#!/bin/sh
# Build target/Kamen.app: a macOS app bundle, so the system (camera permission,
# OBS window capture, the Dock) treats Kamen as a real app.
set -eu
cd "$(dirname "$0")/.."

cargo build --release

APP=target/Kamen.app
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/models"
cp target/release/kamen "$APP/Contents/MacOS/kamen"
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
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSAppSleepDisabled</key><true/>
    <key>NSCameraUsageDescription</key><string>Kamen tracks your face with the camera to animate your avatar.</string>
</dict>
</plist>
PLIST

# Ad-hoc signature: enough to run locally and to keep the camera permission
# between launches. Distribution needs a Developer ID and notarization.
codesign --force --sign - "$APP"
echo "Built $APP"
