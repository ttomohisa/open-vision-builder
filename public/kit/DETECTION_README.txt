Open Vision Builder — local Python D-FINE detection example

Developer investigation only
----------------------------
This kit demonstrates a bounded final-head training/export path. It is not a
qualified detector product. Inspect detectorQualification in manifest.json and
report.json: the current native-prefix feature comparison remains FAIL even
when final logits/boxes and the standalone ONNX comparisons pass. No detection
accuracy, custom-class support, full fine-tuning, or edge-device claim is made.
All 80 original COCO classes remain in their original order.

Install the inference dependencies:
  pip install numpy Pillow onnxruntime

Keep the complete exported model.onnx, labels.json, and detect.py together:
  python detect.py --model model.onnx --image image.png --labels labels.json

Optional output and selection controls:
  python detect.py --model model.onnx --image image.png --labels labels.json \
    --threshold 0.3 --max-detections 100 --output detections.json

The command prints JSON. --output also writes the same JSON to the named file.
Only local files are read. The model contains all inference weights; no PyTorch,
Transformers, source checkpoint, model download, or conversion is required.

Input contract
--------------
The model accepts one FP32 input named images with shape [1,3,640,640].
Apply EXIF orientation first. Decode to RGBA and composite on white with
floating-point alpha. Stretch the complete image to 640 x 640 with half-pixel
bilinear interpolation and edge replication, then divide RGB by 255 and arrange
it as NCHW. There is no padding, crop, or ImageNet mean/std normalization.
Compositing and resizing do not round through an intermediate 8-bit image.
Embedded ICC profiles do not trigger color-profile conversion.

Image dimensions and returned coordinates describe the decoded image after
EXIF orientation has been applied. Browser/Python runtime parity is verified
on an identical procedural 640 x 640 FP32 input. That result does not establish
pixel equality for every image codec or resized image.

Output and selection contract
-----------------------------
The model produces two FP32 tensors:
  logits [1,300,80]: raw independent-class logits for each query.
  boxes  [1,300,4]:  normalized center-x, center-y, width, height.

labels.json is an array of exactly 80 nonempty strings in model-output order.
The script applies sigmoid to each of the 24,000 query/class logits, ranks all
query/class pairs together by descending score, and returns up to 100 pairs
whose score is at least 0.3. Change those defaults with the flags above.
The sigmoid scores do not sum to one. A single query can appear with multiple
class labels. Equal scores are ordered by query index, then class index.
There is no NMS, per-query argmax, or softmax.

Each detection contains queryIndex, classIndex, label, score, and box.
box is [x_min,y_min,x_max,y_max] in pixels of the oriented original image.
Horizontal coordinates are scaled by its width, and vertical coordinates by
its height. Coordinates are not clipped and can extend outside the image.
The JSON also records image width/height and the selection/coordinate contract.

Use the matching labels and preprocessing recipe with this model. Keep its
LICENSE.txt, TRANSFORMERS_LICENSE.txt, NOTICE.txt, MODEL_CARD.md, manifest, and
verification report when distributing the exported kit.
