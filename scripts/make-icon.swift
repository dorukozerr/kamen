// Draws Kamen's app icon (a white mask on a black tile) and writes a 1024 px PNG.
// Run through scripts/make-icon.sh, which turns it into assets/AppIcon.icns.
import AppKit

let side = 1024
let bitmap = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: side, pixelsHigh: side, bitsPerSample: 8, samplesPerPixel: 4,
    hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)

// macOS icon grid: an 824 pt rounded square centered on the 1024 pt canvas.
let tile = NSRect(x: 100, y: 100, width: 824, height: 824)
let tilePath = NSBezierPath(roundedRect: tile, xRadius: 186, yRadius: 186)
NSGradient(
    starting: NSColor(srgbRed: 0.16, green: 0.16, blue: 0.18, alpha: 1),
    ending: NSColor(srgbRed: 0.03, green: 0.03, blue: 0.04, alpha: 1))!
    .draw(in: tilePath, angle: -90)

// The mask: wide at the brow, narrowing to the chin.
let mask = NSBezierPath()
mask.move(to: NSPoint(x: 512, y: 780))
mask.curve(to: NSPoint(x: 770, y: 600), controlPoint1: NSPoint(x: 660, y: 780), controlPoint2: NSPoint(x: 770, y: 720))
mask.curve(to: NSPoint(x: 512, y: 250), controlPoint1: NSPoint(x: 770, y: 430), controlPoint2: NSPoint(x: 640, y: 250))
mask.curve(to: NSPoint(x: 254, y: 600), controlPoint1: NSPoint(x: 384, y: 250), controlPoint2: NSPoint(x: 254, y: 430))
mask.curve(to: NSPoint(x: 512, y: 780), controlPoint1: NSPoint(x: 254, y: 720), controlPoint2: NSPoint(x: 364, y: 780))
mask.close()
let shadow = NSShadow()
shadow.shadowColor = NSColor(white: 0, alpha: 0.5)
shadow.shadowBlurRadius = 36
shadow.shadowOffset = NSSize(width: 0, height: -16)
NSGraphicsContext.saveGraphicsState()
shadow.set()
NSColor(white: 0.97, alpha: 1).setFill()
mask.fill()
NSGraphicsContext.restoreGraphicsState()

// Almond eyes, tilted up at the outer corners.
let eyeColor = NSColor(srgbRed: 0.05, green: 0.05, blue: 0.06, alpha: 1)
for direction in [-1.0, 1.0] {
    let eye = NSBezierPath()
    eye.move(to: NSPoint(x: -72, y: 0))
    eye.curve(to: NSPoint(x: 72, y: 0), controlPoint1: NSPoint(x: -30, y: 44), controlPoint2: NSPoint(x: 30, y: 44))
    eye.curve(to: NSPoint(x: -72, y: 0), controlPoint1: NSPoint(x: 30, y: -30), controlPoint2: NSPoint(x: -30, y: -30))
    var transform = AffineTransform(translationByX: 512 + direction * 112, byY: 585)
    transform.rotate(byDegrees: direction * 12)
    eye.transform(using: transform)
    eyeColor.setFill()
    eye.fill()
}

// A small, calm mouth.
let mouth = NSBezierPath()
mouth.move(to: NSPoint(x: 462, y: 400))
mouth.curve(to: NSPoint(x: 562, y: 400), controlPoint1: NSPoint(x: 490, y: 376), controlPoint2: NSPoint(x: 534, y: 376))
mouth.lineWidth = 16
mouth.lineCapStyle = .round
eyeColor.setStroke()
mouth.stroke()

NSGraphicsContext.current?.flushGraphics()
try! bitmap.representation(using: .png, properties: [:])!
    .write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
