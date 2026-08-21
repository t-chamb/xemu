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
var selfTest = false
var texturesDir: String? = nil

var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--max-dim":
        maxDim = Int(args.removeFirst()) ?? 2048
    case "--force":
        force = true
    case "--self-test":
        selfTest = true
    default:
        texturesDir = (a as NSString).expandingTildeInPath
    }
}

if selfTest {
    texturesDir = NSTemporaryDirectory() + "/texrep-selftest"
    force = true
}

guard let root = texturesDir else {
    FileHandle.standardError.write("usage: texrep-upscale [--max-dim N] [--force] [--self-test] <textures-dir>\n".data(using: .utf8)!)
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

/* Fill the scaler's BGRA input from straight (non-premultiplied) pixels.
 * Deliberately not a CGContext draw over the pixel buffer: CG contexts
 * backed by IOSurface memory were observed to premultiply even through
 * an alpha-stripped source image, silently blackening color under
 * transparent texels. The plain-buffer decode path is verified correct,
 * so route through it and copy. */
func draw(_ image: CGImage, into pb: CVPixelBuffer) {
    let px = straightPixels(image)
    CVPixelBufferLockBaseAddress(pb, [])
    defer { CVPixelBufferUnlockBaseAddress(pb, []) }
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
    let stride = CVPixelBufferGetBytesPerRow(pb)
    let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
    for y in 0..<h {
        for x in 0..<w {
            let d = y * stride + x * 4
            let s = (y * w + x) * 4
            base[d + 0] = px[s + 2]   // B
            base[d + 1] = px[s + 1]   // G
            base[d + 2] = px[s + 0]   // R
            base[d + 3] = 255
        }
    }
}

func cgImage(from pb: CVPixelBuffer) -> CGImage? {
    var img: CGImage?
    VTCreateCGImageFromCVPixelBuffer(pb, options: nil, imageOut: &img)
    return img
}

/* Bicubic-upscale the source's alpha channel to target size, returning a
 * byte per pixel. */
func upscaledAlpha(_ image: CGImage, width: Int, height: Int) -> [UInt8] {
    guard image.alphaInfo != .none && image.alphaInfo != .noneSkipFirst &&
          image.alphaInfo != .noneSkipLast else {
        return [UInt8](repeating: 255, count: width * height)
    }
    /* Must start at zero: drawing composites source-over, and an opaque
     * destination would absorb every source alpha into 255. */
    var alpha = [UInt8](repeating: 0, count: width * height)
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
func assembleStraightRGBA(from pb: CVPixelBuffer, alpha: [UInt8]) -> [UInt8] {
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
    return rgba
}

func straightRGBAImage(_ rgba: [UInt8], width: Int, height: Int) -> CGImage? {
    guard let provider = CGDataProvider(data: Data(rgba) as CFData) else {
        return nil
    }
    return CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
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

/* Per-channel means plus the mean color of transparent texels (the game
 * stores meaningful color there; premultiplication damage shows up as this
 * collapsing toward zero). */
struct Stats {
    var mean = [Double](repeating: 0, count: 4)
    var transMeanRGB = 0.0
    var transCount = 0
}

func stats(rgba: [UInt8], count: Int) -> Stats {
    var s = Stats()
    var sum = [Double](repeating: 0, count: 4)
    var tsum = 0.0
    for i in 0..<count {
        for c in 0..<4 { sum[c] += Double(rgba[i * 4 + c]) }
        if rgba[i * 4 + 3] < 32 {
            s.transCount += 1
            tsum += (Double(rgba[i * 4]) + Double(rgba[i * 4 + 1]) +
                     Double(rgba[i * 4 + 2])) / 3
        }
    }
    for c in 0..<4 { s.mean[c] = sum[c] / Double(count) }
    s.transMeanRGB = s.transCount > 0 ? tsum / Double(s.transCount) : 0
    return s
}

func straightPixels(_ image: CGImage) -> [UInt8] {
    let w = image.width, h = image.height
    var rgb = [UInt8](repeating: 0, count: w * h * 4)
    rgb.withUnsafeMutableBytes { buf in
        let ctx = CGContext(
            data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.interpolationQuality = .none
        ctx.draw(opaqueView(of: image), in: CGRect(x: 0, y: 0, width: w, height: h))
    }
    let alpha = upscaledAlpha(image, width: w, height: h)
    for i in 0..<(w * h) { rgb[i * 4 + 3] = alpha[i] }
    return rgb
}

/* An upscale must not materially change image statistics. Both failure
 * modes seen in practice — premultiplied color collapsing to black, and
 * alpha collapsing to opaque — violate these bounds by an order of
 * magnitude, while legitimate scaler output stays well inside them. */
func verify(source: CGImage, output rgba: [UInt8], outCount: Int,
            name: String) -> Bool {
    let src = stats(rgba: straightPixels(source),
                    count: source.width * source.height)
    let out = stats(rgba: rgba, count: outCount)
    for c in 0..<4 where abs(src.mean[c] - out.mean[c]) > 16 {
        FileHandle.standardError.write(
            "VERIFY FAIL \(name): channel \(c) mean \(Int(src.mean[c])) -> \(Int(out.mean[c]))\n".data(using: .utf8)!)
        return false
    }
    if src.transCount > 100 && src.transMeanRGB > 48 &&
       out.transMeanRGB < src.transMeanRGB * 0.5 {
        FileHandle.standardError.write(
            "VERIFY FAIL \(name): color under transparency \(Int(src.transMeanRGB)) -> \(Int(out.transMeanRGB)) (premultiply damage)\n".data(using: .utf8)!)
        return false
    }
    return true
}

/* Synthetic fixtures modeling the failure modes seen in practice:
 * opaque art, a glyph sheet (bright color everywhere, sparse alpha —
 * catches both premultiply-to-black and alpha-collapse-to-opaque), and a
 * smooth alpha ramp. Exercises the full path including the real scaler. */
if selfTest {
    try? fm.removeItem(atPath: root)
    try? fm.createDirectory(atPath: dumpDir, withIntermediateDirectories: true)
    try? fm.createDirectory(atPath: replaceDir, withIntermediateDirectories: true)
    let n = 128
    var fixtures: [String: [UInt8]] = [:]

    var gradient = [UInt8](repeating: 0, count: n * n * 4)
    var glyphs = [UInt8](repeating: 0, count: n * n * 4)
    var ramp = [UInt8](repeating: 0, count: n * n * 4)
    for y in 0..<n {
        for x in 0..<n {
            let i = (y * n + x) * 4
            gradient[i] = UInt8(x * 2); gradient[i+1] = UInt8(y * 2)
            gradient[i+2] = 128; gradient[i+3] = 255
            glyphs[i] = 230; glyphs[i+1] = 220; glyphs[i+2] = 210
            glyphs[i+3] = ((x / 16 + y / 16) % 2 == 0 && x % 16 < 10 &&
                           y % 16 < 10) ? 255 : 0
            ramp[i] = 200; ramp[i+1] = 64; ramp[i+2] = UInt8(y * 2 % 256)
            ramp[i+3] = UInt8(x * 2 % 256)
        }
    }
    fixtures["fixture-opaque-gradient"] = gradient
    fixtures["fixture-glyph-sheet"] = glyphs
    fixtures["fixture-alpha-ramp"] = ramp

    for (name, rgba) in fixtures {
        guard let img = straightRGBAImage(rgba, width: n, height: n),
              writePNG(img, to: "\(dumpDir)/\(name).png") else {
            FileHandle.standardError.write("fixture write failed: \(name)\n".data(using: .utf8)!)
            exit(1)
        }
    }
    print("self-test: \(fixtures.count) fixtures in \(root)")
}

let entries = (try? fm.contentsOfDirectory(atPath: dumpDir))?
    .filter { $0.hasSuffix(".png") }.sorted() ?? []

if entries.isEmpty {
    print("no dumps found in \(dumpDir)")
    exit(0)
}

var scalers: [String: Scaler] = [:]
var done = 0, skippedBig = 0, skippedExisting = 0, failed = 0
var verifyFailed = 0
let start = Date()

for name in entries {
    let outPath = replaceDir + "/" + name
    if !force && fm.fileExists(atPath: outPath) {
        skippedExisting += 1
        continue
    }
    guard let image = loadCGImage(dumpDir + "/" + name) else {
        FileHandle.standardError.write("FAIL \(name): PNG load\n".data(using: .utf8)!)
        failed += 1
        continue
    }
    let w = image.width, h = image.height
    /* The scaler only does 4x and image inputs cap at 1920; anything whose
     * 4x result exceeds maxDim gets downsampled after upscaling — a 1024
     * source still comes out 2x sharper at the 2048 cap. */
    if w > 1920 || h > 1920 {
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

    guard let src = makePixelBuffer(width: w, height: h) else {
        FileHandle.standardError.write("FAIL \(name): input pixel buffer\n".data(using: .utf8)!)
        failed += 1
        continue
    }
    draw(image, into: src)
    guard let dst = scaler.upscale(src) else {
        FileHandle.standardError.write("FAIL \(name): upscale\n".data(using: .utf8)!)
        failed += 1
        continue
    }

    let alpha = upscaledAlpha(image, width: w * SCALE, height: h * SCALE)
    var rgba = assembleStraightRGBA(from: dst, alpha: alpha)
    var outW = w * SCALE, outH = h * SCALE

    if outW > maxDim || outH > maxDim {
        let s = Double(maxDim) / Double(max(outW, outH))
        let tw = max(Int(Double(outW) * s) & ~1, 2)
        let th = max(Int(Double(outH) * s) & ~1, 2)
        guard let big = straightRGBAImage(rgba, width: outW, height: outH) else {
            FileHandle.standardError.write("FAIL \(name): downsample stage\n".data(using: .utf8)!)
            failed += 1
            continue
        }
        /* Same premultiply-safe split as everywhere else: RGB through the
         * opaque reinterpretation, alpha through its own channel. */
        var down = [UInt8](repeating: 0, count: tw * th * 4)
        down.withUnsafeMutableBytes { buf in
            let ctx = CGContext(
                data: buf.baseAddress, width: tw, height: th,
                bitsPerComponent: 8, bytesPerRow: tw * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            ctx.interpolationQuality = .high
            ctx.draw(opaqueView(of: big), in: CGRect(x: 0, y: 0, width: tw, height: th))
        }
        let downAlpha = upscaledAlpha(big, width: tw, height: th)
        for i in 0..<(tw * th) { down[i * 4 + 3] = downAlpha[i] }
        rgba = down
        outW = tw
        outH = th
    }

    guard verify(source: image, output: rgba,
                 outCount: outW * outH, name: name) else {
        verifyFailed += 1
        continue
    }
    guard let out = straightRGBAImage(rgba, width: outW, height: outH),
          writePNG(out, to: outPath) else {
        FileHandle.standardError.write("FAIL \(name): output write\n".data(using: .utf8)!)
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
             "%d too large (>%d), %d failed, %d rejected by verification",
             done, dt, skippedExisting, skippedBig, maxDim, failed,
             verifyFailed))

if selfTest {
    if done == 3 && failed == 0 && verifyFailed == 0 {
        print("self-test PASS")
        exit(0)
    }
    print("self-test FAIL")
    exit(1)
}
if verifyFailed > 0 {
    exit(1)
}
