/*
 * texrep-upscale — batch AI upscaler for the xemu texture pipeline
 *
 * Walks a texture dump directory (textures/dump/<hash>.png), runs each
 * image through VideoToolbox's super-resolution scaler (Apple Neural
 * Engine, 4x), and writes results into the replacement directory
 * (textures/replace/<hash>.png) consumed by xemu at texture-upload time.
 *
 * The scaler operates on RGB; the alpha channel is upscaled separately
 * with bicubic interpolation and recombined, so UI textures keep clean
 * edges.
 *
 * Usage:
 *   swiftc -O -o texrep-upscale texrep-upscale.swift
 *   ./texrep-upscale [--max-dim 2048] [--force] <textures-dir>
 *
 * <textures-dir> is the directory containing dump/ and replace/
 * (e.g. "~/Library/Application Support/xemu/xemu/textures").
 */

import Foundation
import VideoToolbox
import CoreVideo
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

let SCALE = 4

var maxDim = 2048
var force = false
var texturesDir: String? = nil

var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--max-dim":
        maxDim = Int(args.removeFirst()) ?? 2048
    case "--force":
        force = true
    default:
        texturesDir = (a as NSString).expandingTildeInPath
    }
}

guard let root = texturesDir else {
    FileHandle.standardError.write("usage: texrep-upscale [--max-dim N] [--force] <textures-dir>\n".data(using: .utf8)!)
    exit(2)
}

let dumpDir = root + "/dump"
let replaceDir = root + "/replace"

guard #available(macOS 26.0, *) else {
    FileHandle.standardError.write("VTSuperResolutionScaler requires macOS 26+\n".data(using: .utf8)!)
    exit(1)
}
guard VTSuperResolutionScalerConfiguration.isSupported else {
    FileHandle.standardError.write("Super-resolution scaler not supported on this machine\n".data(using: .utf8)!)
    exit(1)
}

let fm = FileManager.default
try? fm.createDirectory(atPath: replaceDir, withIntermediateDirectories: true)

func loadCGImage(_ path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else {
        return nil
    }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func writePNG(_ image: CGImage, to path: String) -> Bool {
    guard let dst = CGImageDestinationCreateWithURL(
        URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        return false
    }
    CGImageDestinationAddImage(dst, image, nil)
    return CGImageDestinationFinalize(dst)
}

func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
    var pb: CVPixelBuffer?
    let attrs: [CFString: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        kCVPixelBufferCGImageCompatibilityKey: true,
    ]
    CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                        kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
    return pb
}

/* Reinterpret an image's pixels as opaque, so drawing preserves the raw
 * color channels instead of premultiplying them by alpha. Textures store
 * meaningful color under transparent regions (glyph sheets especially);
 * premultiplication would turn those regions black and the scaler would
 * bake dark halos into every edge. */
func opaqueView(of image: CGImage) -> CGImage {
    guard image.bitsPerPixel == 32, let provider = image.dataProvider else {
        return image
    }
    let alphaInfo = image.alphaInfo
    guard alphaInfo != .none && alphaInfo != .noneSkipFirst &&
          alphaInfo != .noneSkipLast else {
        return image
    }
    let skip: CGImageAlphaInfo =
        (alphaInfo == .premultipliedFirst || alphaInfo == .first) ?
            .noneSkipFirst : .noneSkipLast
    let info = CGBitmapInfo(rawValue:
        (image.bitmapInfo.rawValue & ~CGBitmapInfo.alphaInfoMask.rawValue) |
        skip.rawValue)
    return CGImage(
        width: image.width, height: image.height,
        bitsPerComponent: image.bitsPerComponent,
        bitsPerPixel: image.bitsPerPixel, bytesPerRow: image.bytesPerRow,
        space: image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: info, provider: provider, decode: nil,
        shouldInterpolate: false, intent: .defaultIntent) ?? image
}

func draw(_ image: CGImage, into pb: CVPixelBuffer) {
    CVPixelBufferLockBaseAddress(pb, [])
    defer { CVPixelBufferUnlockBaseAddress(pb, []) }
    let ctx = CGContext(
        data: CVPixelBufferGetBaseAddress(pb),
        width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb),
        bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue |
                    CGBitmapInfo.byteOrder32Little.rawValue)!
    ctx.interpolationQuality = .none
    ctx.draw(opaqueView(of: image),
             in: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pb),
                        height: CVPixelBufferGetHeight(pb)))
}

func cgImage(from pb: CVPixelBuffer) -> CGImage? {
    var img: CGImage?
    VTCreateCGImageFromCVPixelBuffer(pb, options: nil, imageOut: &img)
    return img
}

/* Bicubic-upscale the source's alpha channel to target size, returning a
 * byte per pixel. */
func upscaledAlpha(_ image: CGImage, width: Int, height: Int) -> [UInt8] {
    var alpha = [UInt8](repeating: 255, count: width * height)
    guard image.alphaInfo != .none && image.alphaInfo != .noneSkipFirst &&
          image.alphaInfo != .noneSkipLast else {
        return alpha
    }
    alpha.withUnsafeMutableBytes { buf in
        let ctx = CGContext(
            data: buf.baseAddress, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.alphaOnly.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    }
    return alpha
}

/* Combine the scaler's BGRA output with a separately-upscaled straight
 * alpha channel into a straight-alpha RGBA image, bypassing any
 * premultiply/unpremultiply round trip. */
func assembleStraightRGBA(from pb: CVPixelBuffer, alpha: [UInt8]) -> CGImage? {
    CVPixelBufferLockBaseAddress(pb, [])
    defer { CVPixelBufferUnlockBaseAddress(pb, []) }
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
    let stride = CVPixelBufferGetBytesPerRow(pb)
    let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    for y in 0..<h {
        for x in 0..<w {
            let s = y * stride + x * 4
            let d = (y * w + x) * 4
            rgba[d + 0] = base[s + 2]   // B G R A -> R
            rgba[d + 1] = base[s + 1]
            rgba[d + 2] = base[s + 0]
            rgba[d + 3] = alpha[y * w + x]
        }
    }
    guard let provider = CGDataProvider(data: Data(rgba) as CFData) else {
        return nil
    }
    return CGImage(
        width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false,
        intent: .defaultIntent)
}

@available(macOS 26.0, *)
final class Scaler {
    let processor: VTFrameProcessor
    let width: Int, height: Int

    init?(width: Int, height: Int) {
        self.width = width
        self.height = height
        let config = VTSuperResolutionScalerConfiguration(
            frameWidth: width, frameHeight: height,
            scaleFactor: SCALE, inputType: .image,
            usePrecomputedFlow: false,
            qualityPrioritization: .normal,
            revision: .revision1)
        guard let config else { return nil }
        if config.configurationModelStatus ==
            VTSuperResolutionScalerConfiguration.ModelStatus.downloadRequired {
            print("downloading super-resolution model assets…")
            let sem = DispatchSemaphore(value: 0)
            var derr: Error?
            config.downloadConfigurationModel { err in
                derr = err
                sem.signal()
            }
            sem.wait()
            if let derr {
                FileHandle.standardError.write("model download failed: \(derr)\n".data(using: .utf8)!)
                return nil
            }
        }
        processor = VTFrameProcessor()
        do {
            try processor.startSession(configuration: config)
        } catch {
            FileHandle.standardError.write("startSession failed: \(error)\n".data(using: .utf8)!)
            return nil
        }
    }

    deinit {
        processor.endSession()
    }

    func upscale(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        guard let dst = makePixelBuffer(width: width * SCALE,
                                        height: height * SCALE) else {
            return nil
        }
        let srcFrame = VTFrameProcessorFrame(buffer: src,
                                             presentationTimeStamp: .zero)
        let dstFrame = VTFrameProcessorFrame(buffer: dst,
                                             presentationTimeStamp: .zero)
        guard let srcFrame, let dstFrame else { return nil }
        let params = VTSuperResolutionScalerParameters(
            sourceFrame: srcFrame, previousFrame: nil,
            previousOutputFrame: nil, opticalFlow: nil,
            submissionMode: .sequential, destinationFrame: dstFrame)
        guard let params else { return nil }
        let sem = DispatchSemaphore(value: 0)
        var perr: Error?
        processor.process(parameters: params) { _, err in
            perr = err
            sem.signal()
        }
        sem.wait()
        if let perr {
            FileHandle.standardError.write("process failed: \(perr)\n".data(using: .utf8)!)
            return nil
        }
        return dst
    }
}

let entries = (try? fm.contentsOfDirectory(atPath: dumpDir))?
    .filter { $0.hasSuffix(".png") }.sorted() ?? []

if entries.isEmpty {
    print("no dumps found in \(dumpDir)")
    exit(0)
}

var scalers: [String: Scaler] = [:]
var done = 0, skippedBig = 0, skippedExisting = 0, failed = 0
let start = Date()

for name in entries {
    let outPath = replaceDir + "/" + name
    if !force && fm.fileExists(atPath: outPath) {
        skippedExisting += 1
        continue
    }
    guard let image = loadCGImage(dumpDir + "/" + name) else {
        failed += 1
        continue
    }
    let w = image.width, h = image.height
    if w * SCALE > maxDim || h * SCALE > maxDim {
        skippedBig += 1
        continue
    }

    let sizeKey = "\(w)x\(h)"
    if scalers[sizeKey] == nil {
        guard let s = Scaler(width: w, height: h) else {
            FileHandle.standardError.write("cannot create \(sizeKey) scaler\n".data(using: .utf8)!)
            failed += 1
            continue
        }
        scalers[sizeKey] = s
    }
    let scaler = scalers[sizeKey]!

    guard let src = makePixelBuffer(width: w, height: h) else { failed += 1; continue }
    draw(image, into: src)
    guard let dst = scaler.upscale(src) else { failed += 1; continue }

    let alpha = upscaledAlpha(image, width: w * SCALE, height: h * SCALE)
    guard let out = assembleStraightRGBA(from: dst, alpha: alpha),
          writePNG(out, to: outPath) else {
        failed += 1
        continue
    }
    done += 1
    if done % 25 == 0 {
        print("\(done) upscaled…")
    }
}

let dt = Date().timeIntervalSince(start)
print(String(format: "done: %d upscaled in %.1fs, %d already present, " +
             "%d too large (>%d), %d failed",
             done, dt, skippedExisting, skippedBig, maxDim, failed))
