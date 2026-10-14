# MediaPipe face models

Google's MediaPipe Face Landmarker models (Apache-2.0, see `LICENSE`), converted from TensorFlow Lite to ONNX so they run in ONNX Runtime. The weights are unchanged.

| File | What it does | Input | Output |
|---|---|---|---|
| `face_detector.onnx` | BlazeFace short-range: finds the face | 128x128 RGB, -1..1 | 896 boxes with 6 key points, and scores |
| `face_landmarks_detector.onnx` | 478 3D face landmarks, including irises | 256x256 RGB crop, 0..1 | 478 x (x, y, z), face presence |
| `face_blendshapes.onnx` | 52 expression values (ARKit names) | 146 landmarks, x and y in pixels | 52 values, 0..1 |

Source: `face_landmarker.task` from
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`.

`scripts/build.sh models` downloads that bundle and repeats the conversion. Verify the files here with `shasum -a 256 -c SHA256SUMS`.
