# MediaPipe models

Google's MediaPipe Face, Hand and Pose Landmarker models (Apache-2.0, see `LICENSE`), converted from TensorFlow Lite to ONNX so they run in ONNX Runtime. The weights are unchanged; float16 weights become float32, so the files are about twice the size of Google's.

| File | What it does | Input | Output |
|---|---|---|---|
| `face_detector.onnx` | BlazeFace short-range: finds the face | 128x128 RGB, -1..1 | 896 boxes with 6 key points, and scores |
| `face_landmarks_detector.onnx` | 478 3D face landmarks, including irises | 256x256 RGB crop, 0..1 | 478 x (x, y, z), face presence |
| `face_blendshapes.onnx` | 52 expression values (ARKit names) | 146 landmarks, x and y in pixels | 52 values, 0..1 |
| `pose_landmarks_detector.onnx` | 33 body landmarks (lite model) | 256x256 RGB crop, 0..1 | 39 x (x, y, z, visibility, presence), pose presence, 39 x (x, y, z) in meters |
| `hand_landmarks_detector.onnx` | 21 hand landmarks | 224x224 RGB crop, 0..1 | 21 x (x, y, z), hand presence, handedness, 21 x (x, y, z) in meters |

Sources, from `https://storage.googleapis.com/mediapipe-models/`:
`face_landmarker/face_landmarker/float16/latest/face_landmarker.task`,
`hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
`pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`.

Not included: the pose detector (its sparse weights can't be converted with tf2onnx; Kamen places the body crop from the face) and the palm detector (hands are found from the pose's wrists, as in MediaPipe Holistic).

`scripts/build.sh models` downloads the bundles and repeats the conversion. Verify the files here with `shasum -a 256 -c SHA256SUMS`.
