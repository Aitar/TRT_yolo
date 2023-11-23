#include "utils.h"

#define FP4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

#define FP2(ptr) (reinterpret_cast<float2*>(&(ptr))[0])
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

void warpaffine(float* output, uint8_t* origin, float* m, int hi, int wi, int ho, int wo, cudaStream_t stream) {
    dim3 gridSize;
    dim3 blockSize;
    blockSize.x = 32;
    blockSize.y = 32;
    gridSize.x = wo / blockSize.x;
    gridSize.y = ho / blockSize.y;
    warpaffineBinearKernel<<<gridSize, blockSize, 0, stream>>>(origin, output, m, hi, wi, ho, wo, 114);
}

/**
 * 计算confindence和label
 * 启动boxNum个block, 每个block负责对一个box对应的数据处理
 * blockSize = (classNum / 2 / 32 + 1) * 32
 */
__global__ void confidncesAndLabelsKernel(float* res,
                                          const int classNum,
                                          const float confidenceThreshold,
                                          float* labels,
                                          float* confidences) {

    uint ofs = blockIdx.x * (classNum + 5);
    float objness;
    objness = res[ofs + 4];

    if (objness < confidenceThreshold)
        return;

    extern __shared__ float smem[];
    smem[threadIdx.x] = 0.f;
    smem[(blockDim.x << 1) + threadIdx.x] = (float) threadIdx.x;
    smem[blockDim.x * 3 + threadIdx.x] = (float) (threadIdx.x + blockDim.x);

    smem[threadIdx.x] = res[ofs + 5 + threadIdx.x];
    if (threadIdx.x + blockDim.x < classNum)
        smem[blockDim.x + threadIdx.x] = res[ofs + 5 + blockDim.x + threadIdx.x];

    __syncthreads();

#pragma unroll
    for (uint stride = blockDim.x; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && smem[threadIdx.x] < smem[threadIdx.x + stride]) {
            smem[threadIdx.x] = smem[threadIdx.x + stride];
            smem[(blockDim.x << 1) + threadIdx.x] = smem[(blockDim.x << 1) + threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        objness *= smem[0];
        labels[blockIdx.x] = smem[blockDim.x << 1];
        confidences[blockIdx.x] = objness;
    }
    __syncthreads();
}

__global__ void loadBoxKernel(float* res, float* boxes, const int boxNum, const int classNum, float* d2i) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > boxNum) return;

    uint ofs = tid * (classNum + 5);
    float pos[4];
    float box[4];
    for (int i = 0; i < 4; ++i)
        pos[i] = res[ofs + i];

    box[0] = pos[0] - pos[2] * 0.5;
    box[1] = pos[1] - pos[3] * 0.5;
    box[2] = pos[0] + pos[2] * 0.5;
    box[3] = pos[1] + pos[3] * 0.5;

    box[0] = box[0] * d2i[0] + d2i[2];
    box[1] = box[1] * d2i[0] + d2i[5];
    box[2] = box[2] * d2i[0] + d2i[2];
    box[3] = box[3] * d2i[0] + d2i[5];

    FP4(boxes[tid << 2]) = FP4(box[0]);
}

__device__ float calIOU(const float* a, const float* b) {
    float cross_left = max(a[0], b[0]);
    float cross_top = max(a[1], b[1]);
    float cross_right = min(a[2], b[2]);
    float cross_bottom = min(a[3], b[3]);

    float crossArea = max(0.0f, cross_right - cross_left) * max(0.0f, cross_bottom - cross_top);
    float unionArea = max(0.0f, a[2] - a[0]) * max(0.0f, a[3] - a[1])
                       + max(0.0f, b[2] - b[0]) * max(0.0f, b[3] - b[1]) - crossArea;

    if (crossArea == 0 || unionArea == 0) return 0.f;
    return crossArea / unionArea;
}

/**
 * 每个block对一个box进行iou计算, 启动256个线程
 * @param boxes 由上一个核函数整理好的box结果, 包含仿射变换后的方框位置, 为{left, top, right, bottom}的pack
 * @param classNum 
 * @param boxNum 
 */
__global__ void nmsCuda(float* boxes,
                        bool* keep,
                        const float nmsThreshold,
                        const float confThreshold,
                        const int boxNum,
                        const float* confidences) {

    float curConfidence = confidences[blockIdx.x];
    if (curConfidence < confThreshold) {
        if (threadIdx.x == 0)
            keep[blockIdx.x] = false;
        return;
    }

    const uint ofs = blockIdx.x << 2;
    float curBox[4];
    float box[2][4];
    float confidence;

    __shared__ bool isKeepShared[256];
    bool isKeep[2] = {true};
    FP4(curBox[0]) = FP4(boxes[ofs]);

#pragma unroll
    for (uint i = threadIdx.x; i < boxNum; i += blockDim.x) {
        confidence = confidences[i];
        if (confidence > confThreshold) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                box[i & 1][j] = boxes[(i << 2) + j];
            if (i != blockIdx.x && calIOU(curBox, box[i & 1]) > nmsThreshold)
                isKeep[i & 1] &= curConfidence > confidence;
        }
    }

    isKeepShared[threadIdx.x] = isKeep[0] && isKeep[1];
    __syncthreads();

#pragma unroll
    for (uint stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            isKeepShared[threadIdx.x] &= isKeepShared[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        keep[blockIdx.x] = isKeepShared[0];
    }
}

void decoderCuda(float* res,
                 float* labels,
                 float* confidences,
                 float* boxes,
                 float* d2i,
                 bool* isKeep,
                 const int classNum,
                 const int boxNum,
                 cudaStream_t stream,
                 const float confidence_threshold,
                 const float nms_threshold) {

    int blockSize = (classNum / 2 / 32 + 1) * 32;
    confidncesAndLabelsKernel<<<boxNum, blockSize, blockSize << 4, stream>>>(res, classNum, confidence_threshold, labels, confidences);
    loadBoxKernel<<<(boxNum / 256) + 1, 256, 0, stream>>>(res, boxes, boxNum, classNum, d2i);
    nmsCuda<<<boxNum, 256, 0, stream>>>(boxes, isKeep, nms_threshold, confidence_threshold, boxNum, confidences);
}

__global__ void setValueKernel(float* x, float value, int n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint offset = idx * 8;
    if (offset < n) {
        for (int i = 0; i < 8 && offset + i < n; ++i)
            x[offset + i] = value;
    }
}

void setValue(float* x, float value, int n, cudaStream_t stream=nullptr) {
    int blockSize = std::min(512, (n >> 3) + 1);
    int gridSize = n / (blockSize << 3) + 1;
    setValueKernel<<<gridSize, blockSize, 0, stream>>>(x, value, n);
}