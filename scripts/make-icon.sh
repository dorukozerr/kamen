#!/bin/sh
# Regenerate assets/AppIcon.icns from scripts/make-icon.swift. Only needed
# when the icon design changes; the result is committed.
set -eu
cd "$(dirname "$0")/.."

SET=target/AppIcon.iconset
rm -rf "$SET"
mkdir -p "$SET" assets
swift scripts/make-icon.swift assets/AppIcon.png
for size in 16 32 128 256 512; do
    double=$((size * 2))
    sips -z "$size" "$size" assets/AppIcon.png --out "$SET/icon_${size}x${size}.png" >/dev/null
    sips -z "$double" "$double" assets/AppIcon.png --out "$SET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$SET" -o assets/AppIcon.icns
echo "Built assets/AppIcon.icns"
