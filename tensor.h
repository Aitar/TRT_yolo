//
// Created by Aitar on 2022/7/17.
//

#ifndef YOLO_TENSOR_H
#define YOLO_TENSOR_H

#include <opencv2/opencv.hpp>
#include "utils.h"

namespace TRT {
    using namespace std;

    class Tensor {
    public:
        Tensor() = default;

        Tensor(vector<int>& dims) {
            dims_ = make_shared<vector<int>>();
            dims_->assign(dims.begin(), dims.end());
            for (int i : *dims_)
                size_ *= i;
            CUDACHEK(cudaMallocHost((void**)&cpu_, sizeof(float) * size_));
        }

        void setStream(shared_ptr<cudaStream_t>& stream) {
            stream_ = stream;
        }

        shared_ptr<cudaStream_t> getStream() {
            return stream_;
        }

        float* cpu() {
            if (onGPU_) {
//                CUDACHEK(cudaMemcpy(cpu_, gpu_, sizeof(float) * size_, cudaMemcpyHostToDevice));
                CUDACHEK(cudaMemcpyAsync(cpu_, gpu_, sizeof(float) * size_, cudaMemcpyHostToDevice, *stream_));
                onGPU_ = false;
            }
            return cpu_;
        }

        float* gpu() {
            if (gpu_ == nullptr)
                CUDACHEK(cudaMalloc((void**)&gpu_, sizeof(float) * size_));
            if (!onGPU_) {
//                CUDACHEK(cudaMemcpy(gpu_, cpu_, sizeof(float) * size_, cudaMemcpyHostToDevice));
                CUDACHEK(cudaMemcpyAsync(gpu_, cpu_, sizeof(float) * size_, cudaMemcpyHostToDevice, *stream_));
                onGPU_ = true;
            }
            return gpu_;
        }

        bool equals(shared_ptr<Tensor>& tensor) {
            float th = 0.00001;
            for (int i = 0; i < size_; ++i) {
                if (abs(tensor->cpu()[i] - this->cpu()[i]) > th)
                    return false;
            }
            return true;
        }

    public:
        shared_ptr<vector<int>> dims_;
        shared_ptr<cudaStream_t> stream_;
        int size_ = 1;
        float* gpu_ = nullptr;
        float* cpu_ = nullptr;
        bool onGPU_ = false;
    };

    class TensorManager {
    public:
        TensorManager(vector<int>& dims, int size = 10) {
            for (int i = 0; i < size; ++i)
                tensors_.emplace(make_shared<Tensor>(dims));
        }

        shared_ptr<Tensor> allocate() {
            if (tensors_.empty()) return nullptr;
            shared_ptr<Tensor> tensor = tensors_.front();
            tensors_.pop();
            return tensor;
        }

        void back(const shared_ptr<Tensor>& tensor) {
            tensors_.emplace(tensor);
        }

    public:
        queue<shared_ptr<Tensor>> tensors_;
    };
}

#endif //YOLO_TENSOR_H
