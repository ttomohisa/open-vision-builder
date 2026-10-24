Open Vision Builder — local Python classification example

Install the inference dependencies:
  pip install onnxruntime numpy Pillow

Keep the exported model.onnx and labels.json beside predict.py, then run:
  python predict.py --model model.onnx --image image.png --labels labels.json

The command prints JSON with the predicted class, logits, probabilities, and
the top five class probabilities. labels.json is a JSON array of class names
in model-output order. No PyTorch, timm, or conversion step is required.

Input recipe: apply EXIF orientation; decode RGBA; composite on white using
floating-point alpha; stretch the entire image to 224 x 224 with half-pixel
bilinear interpolation; divide RGB by 255; normalize with ImageNet mean
[0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]; use FP32 NCHW.
There is no intermediate 8-bit rounding during compositing or resizing.
ICC color-profile conversion is intentionally not applied. The browser and this
example use different image codecs; fixed PNG, lossless WebP, and JPEG fixtures
are regression-tested, but pixel equality for every image/codec is not promised.
Model-runtime parity is tested separately on the exact same FP32 input tensor.

This application's stretch recipe differs from the original timm model's
bicubic resize and center crop. The original model's benchmark accuracy does
not establish this exported model's accuracy. Evaluate your own held-out data.

Keep LICENSE.txt, NOTICE.txt, MODEL_CARD.md, and the exported manifest/report
with the model when distributing it. Inference reads local files only.
