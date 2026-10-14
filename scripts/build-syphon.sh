#!/bin/sh
# Build target/syphon/libSyphon.dylib from the official Syphon framework
# source (BSD 2-clause), unmodified, at a pinned commit. Syphon lets Kamen send
# its frames straight to OBS, with transparency and without window capture.
#
# The project's prebuilt SDK is Intel-only, so it is compiled here with clang
# (Command Line Tools are enough). The Metal classes are left out: they need
# Xcode's Metal compiler and Kamen draws with OpenGL.
set -eu
cd "$(dirname "$0")/.."

REPO=https://github.com/Syphon/Syphon-Framework
COMMIT=f4761677a45b8034a3c2069ec0f3d2553da81fba
OUT=target/syphon
SRC=$OUT/src

mkdir -p "$OUT"
if [ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null || true)" != "$COMMIT" ]; then
    rm -rf "$SRC"
    git clone -q "$REPO" "$SRC"
    git -C "$SRC" checkout -q "$COMMIT"
fi

# The sources import their own headers as <Syphon/...>.
mkdir -p "$OUT/include"
ln -sfn ../src "$OUT/include/Syphon"

SOURCES=$(ls "$SRC"/*.m "$SRC"/*.c | grep -v Metal)
# shellcheck disable=SC2086
clang -dynamiclib -arch arm64 -arch x86_64 -mmacosx-version-min=11.0 -O2 \
    -fobjc-arc -fobjc-weak -include "$SRC/Syphon_Prefix.pch" \
    -DSYPHON_CORE_SHARE -DGL_SILENCE_DEPRECATION -w -I"$SRC" -I"$OUT/include" \
    $SOURCES \
    -framework Cocoa -framework OpenGL -framework IOSurface \
    -install_name @rpath/libSyphon.dylib \
    -o "$OUT/libSyphon.dylib"
cp "$SRC/License.txt" "$OUT/LICENSE-Syphon.txt"
echo "Built $OUT/libSyphon.dylib"
