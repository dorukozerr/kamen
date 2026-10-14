#!/bin/sh
# Build target/Kamen-<version>.dmg: the app plus a shortcut to Applications,
# for sharing. Without a Developer ID signature and notarization, people who
# download it have to right-click the app and choose Open the first time.
set -eu
cd "$(dirname "$0")/.."

./scripts/bundle-macos.sh

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
STAGE=target/dmg
DMG=target/Kamen-$VERSION.dmg
rm -rf "$STAGE" "$DMG"
mkdir -p "$STAGE"
cp -cR target/Kamen.app "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname Kamen -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"
echo "Built $DMG"
