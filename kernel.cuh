//
// Created by Aitar on 2022/7/19.
//

#ifndef YOLO_KERNEL_CUH
#define YOLO_KERNEL_CUH

#include "utils.h"

#define BGROFFSET(row, col, w) (3 * ((row) * (w) + (col)))

__global__ void warpaffineBinearKernel(uint8_t* origin, float* dst, const float* m, int oh, int ow, int dh, int dw, uint8_t ct) {
    unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dy >= dh || dx >= dw) return;

    float oy = dx * m[3] + dy * m[4] + m[5];
    float ox = dx * m[0] + dy * m[1] + m[2];

    int y = (int)floorf(oy);
    int x = (int)floorf(ox);
    float u = ox - x;
    float v = oy - y;
    float w[] = {(1 - u) * (1 - v), (1 - u) * v, u * (1 - v), u * v};
    uint8_t values[] = {ct, ct, ct};    // B G R

    uint8_t* v0 = &values[0];
    uint8_t* v1 = &values[0];
    uint8_t* v2 = &values[0];
    uint8_t* v3 = &values[0];

    if (x >= 0 && y >= 0 && x < ow && y < oh)
        v0 = origin + BGROFFSET(y, x, ow);
    if (x >= 0 && y + 1 < oh && x < ow && y >= -1)
        v1 = origin + BGROFFSET(y + 1, x, ow);
    if (x + 1 < ow && y >= 0 && x >= -1 && y < oh)
        v2 = origin + BGROFFSET(y, x + 1, ow);
    if (x + 1 < ow && y + 1 < oh && x >= -1 && y >= -1)
        v3 = origin + BGROFFSET(y + 1, x + 1, ow);

    int imgArea = dh * dw;
    int offsetB =               dy * dw + dx;
    int offsetG = imgArea     + dy * dw + dx;
    int offsetR = imgArea * 2 + dy * dw + dx;

    dst[offsetR] = (w[0] * v0[0] + w[1] * v1[0] + w[2] * v2[0] + w[3] * v3[0]) / 255.0f;
    dst[offsetG] = (w[0] * v0[1] + w[1] * v1[1] + w[2] * v2[1] + w[3] * v3[1]) / 255.0f;
    dst[offsetB] = (w[0] * v0[2] + w[1] * v1[2] + w[2] * v2[2] + w[3] * v3[2]) / 255.0f;
}

void warpaffine(const std::shared_ptr<TRT::Tensor>& tensor, uint8_t* origin, cv::Mat& img, float* m, int h, int w, std::shared_ptr<cudaStream_t>& stream) {
    dim3 gridSize;
    dim3 blockSize;
    blockSize.x = 32;
    blockSize.y = 32;
    gridSize.x = w / blockSize.x;
    gridSize.y = h / blockSize.y;

    CUDACHEK(cudaMemcpyAsync(origin, img.data, sizeof(uint8_t) * img.rows * img.cols * 3, cudaMemcpyHostToDevice, *stream));
    warpaffineBinearKernel<<<gridSize, blockSize, 0, *stream>>>(origin, tensor->gpu(), m, img.rows, img.cols, h, w, 114);
}

#endif //YOLO_KERNEL_CUH
