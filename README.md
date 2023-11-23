# TRT yolo

A model inference optimizing project based on Yolo v5s and TensorRT.

Highlights:
- Disconnecting data dependencies between device and host by using asynchronous queue.
- Wrapping Tensor, applying it in advance and binding it to Job class to avoid latency in resource allocation.
- CUDA implementation of affine transformations and NMS.
- Multiple streams inferring.
- 400fps and 100% GPU utilisation ratio.
