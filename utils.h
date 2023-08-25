#ifndef YOLOV5_UTILS_H
#define YOLOV5_UTILS_H

#include <memory>
#include <string>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <fstream>
#include <vector>
#include <tuple>

bool cudaCheck(cudaError_t code, const char *op, const char *file, int line) {
    if (code != cudaSuccess) {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}
#define CUDACHEK(op)  cudaCheck((op), #op, __FILE__, __LINE__)

struct Timer {
public:
    Timer() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void end(const std::string& str) const {
        auto end = std::chrono::high_resolution_clock::now(); // get current time again
        std::chrono::duration<double, std::milli> duration = end - start_; // calculate duration in milliseconds
        std::cout << str << " time: " << duration.count() << " ms" << std::endl; // print duration
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

struct Size{
    int width = 0, height = 0;

    Size() = default;
    Size(int h, int w)
            :width(h), height(w){}
};

struct AffineMatrix{

    float i2d[6];       // image to dst(network), 2x3 matrix
    float d2i[6];       // dst to image, 2x3 matrix
    float* i2dGPU;
    float* d2iGPU;

    void invertAffineTransform(float imat[6], float omat[6]){
        float i00 = imat[0];  float i01 = imat[1];  float i02 = imat[2];
        float i10 = imat[3];  float i11 = imat[4];  float i12 = imat[5];

        float D = i00 * i11 - i01 * i10;
        D = D != 0 ? 1.0 / D : 0;

        float A11 = i11 * D;
        float A22 = i00 * D;
        float A12 = -i01 * D;
        float A21 = -i10 * D;
        float b1 = -A11 * i02 - A12 * i12;
        float b2 = -A21 * i02 - A22 * i12;
        omat[0] = A11;  omat[1] = A12;  omat[2] = b1;
        omat[3] = A21;  omat[4] = A22;  omat[5] = b2;
    }

    void compute(const Size& from, const Size& to){
        float scale = std::min(to.width / (float)from.width, to.height / (float)from.height);

        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;

        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

        invertAffineTransform(i2d, d2i);
        CUDACHEK(cudaMalloc(&d2iGPU, sizeof(float) * 6));
        CUDACHEK(cudaMalloc(&i2dGPU, sizeof(float) * 6));
        CUDACHEK(cudaMemcpy(d2iGPU, &d2i[0], sizeof(float) * 6, cudaMemcpyHostToDevice));
        CUDACHEK(cudaMemcpy(i2dGPU, &i2d[0], sizeof(float) * 6, cudaMemcpyHostToDevice));

    }
};

class CudaManger{
public:
    CudaManger(int size, int memSize, cudaStream_t& outStream) {
        memSize_ = memSize;
        for (int i = 0; i < size; ++i) {
//            void* memPtr;
//            CUDACHEK(cudaMallocAsync(&memPtr, memSize, outStream));
            auto stream = std::make_shared<cudaStream_t>();
            cudaStreamCreate(stream.get());
            streams_.emplace(stream);
            CUDACHEK(cudaStreamSynchronize(outStream));
//            mems_.emplace(memPtr);
        }
    }

    void* getMem(std::shared_ptr<cudaStream_t> stream) {
        void* mem = nullptr;
        if (mems_.empty()){
            CUDACHEK(cudaMalloc(&mem, memSize_));
        } else {
            mem = mems_.front();
            mems_.pop();
        }
        return mem;
    }

    std::shared_ptr<cudaStream_t> getStream() {
        std::shared_ptr<cudaStream_t> stream = nullptr;
        if (streams_.empty()) {
            stream = std::make_shared<cudaStream_t>();
            cudaStreamCreate(stream.get());
        } else {
            stream = streams_.front();
            streams_.pop();
        }
        return stream;
    }

    void returnMem(void* mem) {
        mems_.emplace(mem);
    }

    void returnStream(std::shared_ptr<cudaStream_t>& stream) {
        streams_.emplace(stream);
    }

    bool memEmpty() {
        return mems_.empty();
    }

    int getSize() {
        return memSize_;
    }

private:
    std::queue<void*> mems_;
    std::queue<std::shared_ptr<cudaStream_t>> streams_;
    int memSize_;
};

template<typename T>
static std::shared_ptr <T> nvshared(T *ptr) {
    return std::shared_ptr<T>(ptr, [](T *p) { p->destroy(); });
}



inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

inline bool checkNullptr(void* ptr, const std::string& info) {
    if (ptr == nullptr) {
        printf("%s pointer is null.\n", info.c_str());
        return false;
    }
    return true;
}

static bool saveFile(const std::string& file, void* data, size_t size) {
    std::ofstream f;
    f.open(file, std::ios::out | std::ios::binary);
    if (data && size > 0)
        f.write(reinterpret_cast<const char*> (data), size);
    f.close();
    return true;
}

static std::vector<uint8_t> loadFile(const std::string& file){

    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

// hsv2bgr
static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
        case 0:r = v; g = t; b = p; break;
        case 1:r = q; g = v; b = p; break;
        case 2:r = p; g = v; b = t; break;
        case 3:r = p; g = q; b = v; break;
        case 4:r = t; g = p; b = v; break;
        case 5:r = v; g = p; b = q; break;
        default:r = 1; g = 1; b = 1; break;
    }
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

class TRTLogger: public nvinfer1::ILogger {
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kWARNING){
            if(severity == Severity::kWARNING)
                printf("%s: %s\n", severity_string(severity), msg);
            else if(severity <= Severity::kERROR)
                printf("%s: %s\n", severity_string(severity), msg);
            else
                printf("%s: %s\n", severity_string(severity), msg);
        }
    }
};

static TRTLogger logger;

#endif //YOLOV5_UTILS_H
