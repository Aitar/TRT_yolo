#include <opencv2/opencv.hpp>
#include "utils.h"

namespace TRT {
    using namespace std;

    class Tensor {
    public:
        Tensor() = default;

        Tensor(const vector<int>& dims, cudaStream_t stream = nullptr) {
            dims_ = make_shared<vector<int>>();
            dims_->assign(dims.begin(), dims.end());
            for (int i : *dims_)
                size_ *= i;
            CUDACHEK(cudaMallocHost(&cpu_, sizeof(float) * size_));
            CUDACHEK(cudaMallocAsync(&gpu_, sizeof(float) * size_, stream));
        }

        float* cpu(cudaStream_t stream = nullptr) {
            if (onGPU_) {
                CUDACHEK(cudaMemcpyAsync(cpu_, gpu_, sizeof(float) * size_, cudaMemcpyHostToDevice, stream));
                cudaStreamSynchronize(stream);
                onGPU_ = false;
            }
            return cpu_;
        }

        // 注意cpu -> gpu是没加强制barrier的
        float* gpu(cudaStream_t stream = nullptr) {
            if (!onGPU_) {
                CUDACHEK(cudaMemcpyAsync(gpu_, cpu_, sizeof(float) * size_, cudaMemcpyHostToDevice, stream));
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
        int size_ = 1;
        float* gpu_ = nullptr;
        float* cpu_ = nullptr;
        bool onGPU_ = false;
    };
}