#!/bin/sh
# Build target/Kamen.app: a macOS app bundle with its icon, metadata, tracking
# models and the Syphon library, signed for running on this Mac.
set -eu
cd "$(dirname "$0")/.."

cargo build --release

APP=target/Kamen.app
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
YEAR=$(date +%Y)
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/models"
cp target/release/kamen "$APP/Contents/MacOS/kamen"
cp assets/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
cp LICENSE "$APP/Contents/Resources/LICENSE-Kamen.txt"
printf 'APPL????' > "$APP/Contents/PkgInfo"
# -c clones on APFS, so the large model files take no extra disk space.
cp -cR models/face_landmark "$APP/Contents/Resources/models/"
# The avatar loaded at startup, if there is one.
if [ -d models/vrm ]; then
    cp -cR models/vrm "$APP/Contents/Resources/models/"
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Kamen</string>
    <key>CFBundleDisplayName</key><string>Kamen</string>
    <key>CFBundleIdentifier</key><string>sh.doruk.kamen</string>
    <key>CFBundleExecutable</key><string>kamen</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundleSupportedPlatforms</key><array><string>MacOSX</string></array>
    <key>LSApplicationCategoryType</key><string>public.app-category.video</string>
    <key>NSHumanReadableCopyright</key><string>© $YEAR Doruk Özer. Free software.</string>
    <key>NSPrincipalClass</key><string>NSApplication</string>
    <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSAppSleepDisabled</key><true/>
    <key>NSCameraUsageDescription</key><string>Kamen tracks your face with the camera to animate your avatar.</string>
</dict>
</plist>
PLIST

# Syphon output (optional at run time): build it once, ship it in the bundle.
if [ ! -f target/syphon/libSyphon.dylib ]; then
    ./scripts/build-syphon.sh
fi
mkdir -p "$APP/Contents/Frameworks"
cp target/syphon/libSyphon.dylib "$APP/Contents/Frameworks/"
cp target/syphon/LICENSE-Syphon.txt "$APP/Contents/Resources/"
codesign --force --sign - "$APP/Contents/Frameworks/libSyphon.dylib"

# Ad-hoc signature: enough to run locally and to keep the camera permission
# between launches. Distribution needs a Developer ID and notarization.
codesign --force --sign - "$APP"
echo "Built $APP"
