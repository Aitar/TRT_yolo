#ifndef YOLO_YOLO_H
#define YOLO_YOLO_H


#include "future"
#include <mutex>
#include <utility>
#include <sys/timeb.h>

#include "tensor.h"
#include "context.h"
#include "kernel.cuh"

namespace YOLO {
    using namespace std;
    using namespace TRT;

    static const char *cocolabels[] = {
            "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
            "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    class Box {
    public:
        float left, top, right, bottom, conf;
        int label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float conf, int label) : left(left), top(top),
                                                                                       right(right),
                                                                                       bottom(bottom), conf(conf),
                                                                                       label(label) {}
    };

    typedef std::vector<Box> BoxArray;

    class Job {
    public:
        cv::Mat img_;
        shared_ptr<Tensor> input_;
        shared_ptr<Tensor> output_;
        chrono::time_point<chrono::high_resolution_clock> start_;
        shared_ptr<cudaStream_t> stream_;

        Job() = default;

        Job(cv::Mat &img,
            shared_ptr<Tensor>& input,
            shared_ptr<Tensor>& output,
            chrono::time_point<chrono::high_resolution_clock>& start,
            shared_ptr<cudaStream_t>& stream) :
                img_(std::move(img)),
                input_(std::move(input)),
                output_(std::move(output)),
                start_(start),
                stream_(stream) {};
    };

    class Yolo {
    public:
        Yolo() = default;

        Yolo(
                const string &trtPath,
                const int maxBatchSize,
                const size_t maxWorkspaceSize,
                cudaStream_t &stream,
                Size &imgSize,
//                float* std,
//                float* means,
                const int b = 1,
                const int c = 3,
                const int h = 640,
                const int w = 640,
                const float confTH = 0.25,
                const float nmsTH = 0.5
        ) {
            exeCtx_ = make_shared<EngineContext>(trtPath, stream);
            CUDACHEK(cudaMallocAsync(&mem_, sizeof(uint8_t) * imgSize.height * imgSize.width * 3, stream));
            //printf("Initing...\n");
            imgSize_ = imgSize;
            Size to{h, w};
            am_ = new AffineMatrix();
            am_->compute(imgSize_, to);

            maxBatchSize_ = 16;
            int maxBufferSize = 32;
            maxWorkspace_ = (int) maxWorkspaceSize;
            b_ = b;
            c_ = c;
            h_ = h;
            w_ = w;
            inputSize_ = b * c * h * w;
            vector<int> inputDimsv{b, c, h, w};
            inputTM_ = make_shared<TensorManager>(inputDimsv, 64);
            cudaManger_ = make_shared<CudaManger>(64, sizeof(uint8_t) * imgSize.height * imgSize.width * 3, stream);
            auto outputDims = exeCtx_->engine_->getBindingDimensions(1);
            boxNum_ = outputDims.d[1];
            probNum_ = outputDims.d[2];
            classNum_ = probNum_ - 5;
            outputSize_ = b * boxNum_ * probNum_;
            vector<int> outputDimsv{b, boxNum_, probNum_};
            //printf("inputSize: %dB, outputSize: %dB\n", inputSize_ * 4, outputSize_ * 4);
            outputTM_ = make_shared<TensorManager>(outputDimsv, 64);

            auto inputDims = exeCtx_->engine_->getBindingDimensions(0);
            inputDims.d[0] = b;
            exeCtx_->exeCtx_->setBindingDimensions(0, inputDims);

//            std_ = std;
//            means_ = means;
            confTH_ = confTH;
            nmsTH_ = nmsTH;
            CUDACHEK(cudaStreamSynchronize(stream));
            //printf("\n");
        }

        Yolo(
                const string &src,
                const string &dst,
                const int maxBatchSize,
                const size_t maxWorkspaceSize,
                cudaStream_t stream,
//                float* std,
//                float* means,
                const int b = 1,
                const int c = 3,
                const int h = 640,
                const int w = 640,
                const float confTH = 0.25,
                const float nmsTH = 0.5
        ) {

            exeCtx_ = make_shared<EngineContext>(src, dst, maxBatchSize, maxWorkspaceSize, stream);

            maxBatchSize_ = maxBatchSize;
            maxWorkspace_ = (int) maxWorkspaceSize;
            b_ = b;
            c_ = c;
            h_ = h;
            w_ = w;
            inputSize_ = b * c * h * w;
            vector<int> inputDimsv{b, c, h, w};
            auto outputDims = exeCtx_->engine_->getBindingDimensions(1);
            boxNum_ = outputDims.d[1];
            probNum_ = outputDims.d[2];
            classNum_ = probNum_ - 5;
            outputSize_ = b * boxNum_ * probNum_;
            auto inputDims = exeCtx_->engine_->getBindingDimensions(0);
            inputDims.d[0] = b;
            exeCtx_->exeCtx_->setBindingDimensions(0, inputDims);

            vector<int> outputDimsv{b, boxNum_, probNum_};
            inputTM_ = make_shared<TensorManager>(inputDimsv);
            outputTM_ = make_shared<TensorManager>(outputDimsv);
//            std_ = std;
//            means_ = means;
            confTH_ = confTH;
            nmsTH_ = nmsTH;
        }


        void startup() {
            printf("starting...\n");
            workerThread_ = thread(&Yolo::worker, this);
            resThread_ = thread(&Yolo::receiver, this);
        }

        void preprocess(cv::Mat& img, const shared_ptr<Tensor>& tensor, shared_ptr<cudaStream_t>& stream) {
            CUDACHEK(cudaMemcpyAsync(mem_, img.data, sizeof(uint8_t) * img.cols * img.rows * 3, cudaMemcpyHostToDevice, *stream));
            warpaffine(tensor, mem_, img, am_->d2iGPU, h_, w_, stream);
            CUDACHEK(cudaStreamSynchronize(*stream));
        }

        void commit(cv::Mat &img) {
            chrono::time_point<chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now(); // get current time
            shared_ptr<Tensor> input = nullptr;
            shared_ptr<Tensor> output = nullptr;
            // waiting free tensor
            while (input == nullptr || output == nullptr) {
                unique_lock<mutex> l(tensorLock_);
                //printf("commit locked\n");
                tensorCV_.wait(l, [&]() {
                    return !inputTM_->tensors_.empty() && !outputTM_->tensors_.empty();
                });
                //printf("commit unlocked\n");
                input = inputTM_->allocate();
                output = outputTM_->allocate();
            }

            auto stream = cudaManger_->getStream();
            input->setStream(stream);
            output->setStream(stream);
            preprocess(img, input, stream);

            Job job{img, input, output, start, stream};
            {
                //printf("get lock_\n");
                unique_lock<mutex> l(lock_);
                jobs_.emplace(job);
                cv_.notify_one();
                //printf("release lock_\n");
            }
        }

        void setWriter(const cv::VideoWriter &writer) {
            writer_ = writer;
        }

        void forward(const Job& job) {
            //printf("Inference started\n");
            float *bindings[] = {job.input_->gpu(), job.output_->gpu()};
            bool success = exeCtx_->exeCtx_->enqueueV2((void **) bindings, *job.stream_, nullptr);
            CUDACHEK(cudaStreamSynchronize(*job.stream_));
            //printf("Inference ended\n");
            if (!success)
                printf("Inference failed\n");
        }

        void worker() {
            while (!stop_) {
                {
                    unique_lock<mutex> l(lock_);
                    cv_.wait(l, [&]() {
                        //printf("worker locked\n");
                        return !jobs_.empty() || stop_;
                    });
                    //printf("worker unlocked\n");
                    if (stop_) return;
                    while (!jobs_.empty() && batch_.size() < maxBatchSize_) {
                        batch_.emplace(jobs_.front());
                        jobs_.pop();
                    }
                }
                while (!batch_.empty()) {
                    Job job = batch_.front();
                    batch_.pop();
                    forward(job);
                    {
                        unique_lock<mutex> l(resLock_);
                        results_.emplace(job);
                        resCV_.notify_one();
                    }
                }
            }
        }

        void receiver() {
            vector<Job> jobs;
            while (!stop_) {
                while (!results_.empty()) {
                    unique_lock<mutex> l(resLock_);
                    //printf("receiver locked\n");
                    resCV_.wait(l, [&]() {
                        return !results_.empty() || stop_;
                    });
                    //printf("receiver unlocked\n");
                    if (stop_) return;
                    jobs.emplace_back(results_.front());
                    results_.pop();
                }

                for(const auto & job : jobs)
                    decode(job);

                for(const auto & job : jobs) {
                    //printf("returning, input left: %zu, output left: %zu\n", inputTM_->tensors_.size(), outputTM_->tensors_.size());
                    unique_lock<mutex> l(tensorLock_);
                    inputTM_->back(job.input_);
                    outputTM_->back(job.output_);
                    tensorCV_.notify_one();
                    //printf("returned, input left: %zu, output left: %zu\n", inputTM_->tensors_.size(), outputTM_->tensors_.size());
                }
                jobs.clear();
            }
        }

        void decode(Job job, const string &savePath = "img-draw.jpg") {
            job.output_->cpu();
            vector<vector<float>> bboxes;
            float confidence_threshold = 0.25;
            float nms_threshold = 0.5;
            CUDACHEK(cudaStreamSynchronize(*job.stream_));
            cudaManger_->returnStream(job.stream_);
            for (int i = 0; i < boxNum_; ++i) {
                float *ptr = job.output_->cpu() + i * probNum_;
                float objness = ptr[4];
                if (objness < confidence_threshold)
                    continue;

                float *pclass = ptr + 5;
                int label = std::max_element(pclass, pclass + classNum_) - pclass;
                float prob = pclass[label];
                float confidence = prob * objness;
                if (confidence < confidence_threshold)
                    continue;

                // 中心点、宽、高
                float cx = ptr[0];
                float cy = ptr[1];
                float width = ptr[2];
                float height = ptr[3];

                // 预测框
                float left = cx - width * 0.5;
                float top = cy - height * 0.5;
                float right = cx + width * 0.5;
                float bottom = cy + height * 0.5;

                // 对应图上的位置
                float image_base_left = am_->d2i[0] * left + am_->d2i[2];
                float image_base_right = am_->d2i[0] * right + am_->d2i[2];
                float image_base_top = am_->d2i[0] * top + am_->d2i[5];
                float image_base_bottom = am_->d2i[0] * bottom + am_->d2i[5];
                bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float) label,
                                  confidence});
            }
//            //printf("decoded bboxes.size = %zu\n", bboxes.size());

            // nms非极大抑制
            std::sort(bboxes.begin(), bboxes.end(), [](vector<float> &a, vector<float> &b) { return a[5] > b[5]; });
            std::vector<bool> remove_flags(bboxes.size());
            std::vector<vector<float>> box_result;
            box_result.reserve(bboxes.size());

            auto iou = [](const vector<float> &a, const vector<float> &b) {
                float cross_left = std::max(a[0], b[0]);
                float cross_top = std::max(a[1], b[1]);
                float cross_right = std::min(a[2], b[2]);
                float cross_bottom = std::min(a[3], b[3]);

                float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
                float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1])
                                   + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
                if (cross_area == 0 || union_area == 0) return 0.0f;
                return cross_area / union_area;
            };

            for (int i = 0; i < bboxes.size(); ++i) {
                if (remove_flags[i]) continue;

                auto &ibox = bboxes[i];
                box_result.emplace_back(ibox);
                for (int j = i + 1; j < bboxes.size(); ++j) {
                    if (remove_flags[j]) continue;

                    auto &jbox = bboxes[j];
                    if (ibox[4] == jbox[4]) {
                        // class matched
                        if (iou(ibox, jbox) >= nms_threshold)
                            remove_flags[j] = true;
                    }
                }
            }
//            //printf("box_result.size = %zu\n", box_result.size());

            for (int i = 0; i < box_result.size(); ++i) {
                auto &ibox = box_result[i];
                float left = ibox[0];
                float top = ibox[1];
                float right = ibox[2];
                float bottom = ibox[3];
                int class_label = ibox[4];
                float confidence = ibox[5];
                cv::Scalar color;
                tie(color[0], color[1], color[2]) = random_color(class_label);
                cv::rectangle(job.img_, cv::Point(left, top), cv::Point(right, bottom), color, 3);

                auto name = cocolabels[class_label];
                auto caption = cv::format("%s %.2f", name, confidence);
                int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(job.img_, cv::Point(left - 3, top - 33), cv::Point(left + text_width, top), color, -1);
                cv::putText(job.img_, caption, cv::Point(left, top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
            }

//            cv::imshow("test", job.img_);
//            cv::waitKey(1);
        }

        void stop() {
            stop_ = true;
            cv_.notify_one();
            workerThread_.join();
            resThread_.join();
            writer_.release();
        }

        void setHW(int h, int w) {
            h_ = h;
            w_ = w;
        }

    private:
        int maxBatchSize_;
        int maxWorkspace_;
        int b_, c_, h_, w_, inputSize_;
        int boxNum_, probNum_, classNum_, outputSize_;
        float confTH_;
        float nmsTH_;
        float *std_ = nullptr;
        float *means_ = nullptr;
        uint8_t *mem_;

        AffineMatrix *am_;
        Size imgSize_;

        cv::VideoWriter writer_;
        shared_ptr<CudaManger> cudaManger_ = nullptr;
        shared_ptr<EngineContext> exeCtx_ = nullptr;
        shared_ptr<TensorManager> inputTM_ = nullptr;
        shared_ptr<TensorManager> outputTM_ = nullptr;
        queue<Job> jobs_;
        queue<Job> batch_;
        queue<Job> results_;
        queue<uint8_t *> cudaMems_;

        mutex lock_;
        mutex tensorLock_;
        mutex resLock_;
        mutex memLock_;
        condition_variable cv_;
        condition_variable tensorCV_;
        condition_variable resCV_;
        condition_variable memCV_;
        thread workerThread_;
        thread resThread_;
        bool stop_ = false;
    };
}

#endif //YOLO_YOLO_H
