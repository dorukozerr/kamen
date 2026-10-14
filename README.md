# Kamen

a free, open-source VTuber app for macOS

Bring your VRM avatar to life with just your webcam. Kamen tracks your face,
eyes and expressions, and sends the avatar straight to OBS with a transparent
background through Syphon. No green screen, no subscription, no watermark. Free
and open source. Free VTuber app attempt for macos

## Demo

https://github.com/user-attachments/assets/672f0764-249f-4d83-8662-a17bf0fa40dc

## Build and run it yourself

A signed download is coming. Until then, you can build Kamen from source.

You need a Mac with Apple Silicon, plus:

- Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Xcode Command Line Tools (for `clang` and `git`): `xcode-select --install`

Then:

```sh
git clone https://github.com/dorukozerr/kamen.git
cd kamen
scripts/build
open target/Kamen.app
```

The first build takes a few minutes. The tracking models are in the repo, so nothing else is downloaded except Rust crates and the Syphon source.

macOS asks for camera access on first launch. With the default ad-hoc signature it asks again after every rebuild.

## Using it with OBS

1. In Kamen, keep **Send to OBS** on (it is by default).
2. In OBS, add a **Syphon Client** source and pick Kamen.
3. For a transparent background, choose **Transparent** under Background in Kamen and turn on **Allow Transparency** in the OBS source.

Load your own avatar with **Change…**. Kamen uses VRM models, such as the ones VRoid Studio exports.

## License

Kamen is free software under the WTFPL (see `LICENSE`). The tracking models are Google's MediaPipe models (Apache-2.0, see `models/mediapipe/`), and Syphon is BSD-licensed.
