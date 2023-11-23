#include <future>
#include <mutex>
#include <utility>
#include <sys/timeb.h>

#include "context.h"
#include "kernel.cuh"
#include "tensor.h"

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

    class Job {
    private:
        cv::Mat img_;
        shared_ptr<Tensor> input_ = nullptr;
        shared_ptr<Tensor> output_ = nullptr;
        shared_ptr<Tensor> labels_ = nullptr;
        shared_ptr<Tensor> confidences_ = nullptr;
        shared_ptr<Tensor> boxes_ = nullptr;
        bool *isKeepCPU_ = nullptr;
        bool *isKeepGPU_ = nullptr;
        uint8_t* origin_ = nullptr;

        chrono::time_point<chrono::high_resolution_clock> start_;
        cudaStream_t stream_ = nullptr;

    public:
        Job() = default;

        Job(const shared_ptr<Tensor> &input,
            const shared_ptr<Tensor> &output,
            const shared_ptr<Tensor> &labels,
            const shared_ptr<Tensor> &confidences,
            const shared_ptr<Tensor> &boxes,
            bool *isKeepCpu,
            bool *isKeepGpu,
            uint8_t* origin)
                : input_(input), output_(output), labels_(labels),
                  confidences_(confidences), boxes_(boxes), isKeepCPU_(isKeepCpu),
                  isKeepGPU_(isKeepGpu), origin_(origin) {

            cudaStreamCreate(&stream_);
        }

        ~Job() {
            if (isKeepCPU_ != nullptr) {
                delete (isKeepCPU_);
                isKeepCPU_ = nullptr;
            }

            if (isKeepGPU_ != nullptr) {
                cudaFree(isKeepGPU_);
                isKeepGPU_ = nullptr;
            }

            if (origin_ != nullptr) {
                cudaFree(origin_);
                origin_ = nullptr;
            }

            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }
        }

        uint8_t* getOrigin() const {
            return origin_;
        }

        cv::Mat &getImg() {
            return img_;
        }

        float* getInputGPU() {
            return input_->gpu(stream_);
        }

        float* getOutputGPU() {
            return output_->gpu(stream_);
        }

        float* getInputCPU() {
            return input_->cpu(stream_);
        }

        float* getOutputCPU() {
            return output_->cpu(stream_);
        }

        const shared_ptr<Tensor> &getLabels() const {
            return labels_;
        }

        const shared_ptr<Tensor> &getConfidences() const {
            return confidences_;
        }

        const shared_ptr<Tensor> &getBoxes() const {
            return boxes_;
        }

        bool *getIsKeepCpu() const {
            return isKeepCPU_;
        }

        bool *getIsKeepGpu() const {
            return isKeepGPU_;
        }

        const chrono::time_point<chrono::high_resolution_clock> &getStart() const {
            return start_;
        }

        cudaStream_t getStream() {
            return stream_;
        }

        void setStart(const chrono::time_point<chrono::high_resolution_clock> &start) {
            start_ = start;
        }

        void setImg(const cv::Mat &img) {
            img_ = img;
            cudaMemcpyAsync(origin_, img_.data, sizeof(uint8_t) * img_.cols * img_.rows * 3, cudaMemcpyHostToDevice, stream_);
        }
    };

    class JobManager {
    private:
        list<shared_ptr<Job>> jobList_;

    public:
        JobManager(const vector<int> &inputDims,
                   const vector<int> &outputDims,
                   const int boxNum_,
                   const int ih, const int iw,
                   const int size = 32,
                   cudaStream_t stream = nullptr) {

            for (int i = 0; i < size; ++i) {
                auto input = make_shared<Tensor>(inputDims, stream);
                auto output = make_shared<Tensor>(outputDims, stream);
                auto labels = make_shared<Tensor>(vector<int>{boxNum_}, stream);
                auto confidences = make_shared<Tensor>(vector<int>{boxNum_}, stream);
                auto boxes = make_shared<Tensor>(vector<int>{boxNum_ << 2}, stream);
                bool *isKeepCPU = new bool[boxNum_];
                bool *isKeepGPU = nullptr;
                cudaMallocAsync(&isKeepGPU, sizeof(bool) * boxNum_, stream);
                uint8_t* origin = nullptr;
                cudaMallocAsync(&origin, sizeof(uint8_t) * ih * iw * 3, stream);

                jobList_.emplace_back(
                        make_shared<Job>(input, output, labels, confidences, boxes, isKeepCPU, isKeepGPU, origin));
            }

        }

        shared_ptr<Job> getJob() {
//            printf("get %zu\n", jobList_.size());
            if (!jobList_.empty()) {
                auto job = jobList_.front();
                jobList_.pop_front();
                return job;
            } else return nullptr;
        }

        void returnJob(shared_ptr<Job> job) {
            jobList_.emplace_back(std::move(job));
//            printf("return %zu\n", jobList_.size());
        }

        bool empty() {
            return jobList_.empty();
        }
    };

    class Yolo {
    public:
        Yolo() = default;

        Yolo(const string &trtPath,
             const int maxBatchSize,
             const int maxWorkspaceSize,
             const int ih,
             const int iw,
             const int n = 1,
             const int c = 3,
             const int h = 640,
             const int w = 640,
             const float confTH = 0.25,
             const float nmsTH = 0.5
        ) :
                maxBatchSize_(maxBatchSize), maxWorkspace_(maxWorkspaceSize),
                ih_(ih), iw_(iw), n_(n), c_(c), h_(h), w_(w), inputSize_(n * c * h * w),
                confTH_(confTH), nmsTH_(nmsTH) {
            exeCtx_ = make_shared<EngineContext>(trtPath);
            init();

        }

        Yolo(const string &onnxPath,
             const string &trtPath,
             const int maxBatchSize,
             const int maxWorkspaceSize,
             const int ih,
             const int iw,
             const int n = 1,
             const int c = 3,
             const int h = 640,
             const int w = 640,
             const float confTH = 0.25,
             const float nmsTH = 0.5
        ) :
                maxBatchSize_(maxBatchSize), maxWorkspace_(maxWorkspaceSize),
                ih_(ih), iw_(iw), n_(n), c_(c), h_(h), w_(w), inputSize_(n * c * h * w),
                confTH_(confTH), nmsTH_(nmsTH) {

            exeCtx_ = make_shared<EngineContext>(onnxPath, trtPath, maxBatchSize, maxWorkspaceSize);
            init();
        }

        void init() {
            auto outputDims = exeCtx_->engine_->getBindingDimensions(1);
            auto inputDims = exeCtx_->engine_->getBindingDimensions(0);

            boxNum_ = outputDims.d[1];
            probNum_ = outputDims.d[2];
            classNum_ = probNum_ - 5;
            outputSize_ = n_ * boxNum_ * probNum_;

            inputDims.d[0] = n_;
            exeCtx_->exeCtx_->setBindingDimensions(0, inputDims);

            am_ = new AffineMatrix((float) ih_, (float) iw_, (float) h_, (float) w_);
            jobManager_ = make_shared<JobManager>(vector<int>{n_, c_, h_, w_}, vector<int>{n_, boxNum_, probNum_}, boxNum_, ih_, iw_, maxWorkspace_);
        }

        void setWriter(const cv::VideoWriter &writer) {
            writer_ = writer;
        }

        void startup() {
            printf("starting...\n");
            workerThread_ = thread(&Yolo::worker, this);
            resThread_ = thread(&Yolo::receiver, this);
        }

        void stop() {
            stop_ = true;
            cv_.notify_one();
            workerThread_.join();
            resThread_.join();
            writer_.release();
        }

        void commit(cv::Mat &img) {
            chrono::time_point<chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now(); // get current time
            // waiting for a free job
            shared_ptr<Job> job = nullptr;
            while (job == nullptr) {
                unique_lock<mutex> lock(jobManagerLock_);
                jobCV_.wait(lock, [&]() {
                    return !jobManager_->empty();
                });
                job = jobManager_->getJob();
            }

            job->setStart(start);
            job->setImg(img);

            {
                //printf("get lock_\n");
                unique_lock<mutex> l(jobLock_);
                jobs_.emplace_back(std::move(job));
                cv_.notify_one();
                //printf("release lock_\n");
            }
        }

        void worker() {
            while (!stop_) {
                {
                    unique_lock<mutex> l(jobLock_);
                    cv_.wait(l, [&]() {
                        return !jobs_.empty() || stop_;
                    });
                    if (stop_) return;
                    while (!jobs_.empty() && batch_.size() < maxBatchSize_) {
                        batch_.emplace_back(jobs_.front());
                        jobs_.pop_front();
                    }
                }

                for (const auto& job: batch_)
                    forward(job);

                while (!batch_.empty()) {
                    auto job = batch_.front();
                    batch_.pop_front();
                    {
                        unique_lock<mutex> l(resLock_);
                        results_.emplace(std::move(job));
                        resCV_.notify_one();
                    }
                }
            }
        }

        void receiver() {
            list<shared_ptr<Job>> jobs;
            while (!stop_) {
                while (!results_.empty()) {
                    unique_lock<mutex> l(resLock_);
                    //printf("receiver locked\n");
                    resCV_.wait(l, [&]() {
                        return !results_.empty() || stop_;
                    });
                    //printf("receiver unlocked\n");
                    if (stop_) return;
                    jobs.emplace_back(std::move(results_.front()));
                    results_.pop();
                }

                for(const auto& job : jobs)
                    drawRectrangle(job);

                for(const auto& job : jobs) {
                    //printf("returning, input left: %zu, output left: %zu\n", inputTM_->tensors_.size(), outputTM_->tensors_.size());
                    unique_lock<mutex> lock(jobManagerLock_);
                    jobManager_->returnJob(job);
                    jobCV_.notify_one();
                    //printf("returned, input left: %zu, output left: %zu\n", inputTM_->tensors_.size(), outputTM_->tensors_.size());
                }
                jobs.clear();
            }
        }

        void forward(const shared_ptr<Job>& job) {
            warpaffine(job->getInputGPU(), job->getOrigin(), am_->d2iGPU, ih_, iw_, h_, w_, job->getStream());
            float *bindings[] = {job->getInputGPU(), job->getOutputGPU()};
            bool success = exeCtx_->exeCtx_->enqueueV2((void **) bindings, job->getStream(), nullptr);
            if (!success) {
                printf("Inference failed.\n");
                return;
            }
            //printf("Inference ended\n");
            decoderCuda(job->getOutputGPU(),
                        job->getLabels()->gpu(job->getStream()),
                        job->getConfidences()->gpu(job->getStream()),
                        job->getBoxes()->gpu(job->getStream()),
                        am_->d2iGPU, job->getIsKeepGpu(), classNum_, boxNum_, job->getStream(), confTH_, nmsTH_);
        }

        void drawRectrangle(const shared_ptr<Job>& job) const {
            auto isKeep = job->getIsKeepCpu();
            cudaMemcpyAsync(isKeep, job->getIsKeepGpu(), sizeof(bool) * boxNum_, cudaMemcpyDeviceToHost, job->getStream());
            auto boxes = job->getBoxes()->cpu(job->getStream());
            auto confidences = job->getConfidences()->cpu(job->getStream());
            auto labels = job->getLabels()->cpu(job->getStream());
            cudaStreamSynchronize(job->getStream());

            int ofs;

            for (int i = 0; i < boxNum_; ++i) {
                if (!isKeep[i]) continue;
                ofs = i << 2;
                float left = boxes[ofs];
                float top = boxes[ofs + 1];
                float right = boxes[ofs + 2];
                float bottom = boxes[ofs + 3];
                int class_label = (int) labels[i];
                float confidence = confidences[i];
                cv::Scalar color;
                tie(color[0], color[1], color[2]) = random_color(class_label);
                cv::rectangle(job->getImg(), cv::Point(left, top), cv::Point(right, bottom), color, 3);

                auto name = cocolabels[class_label];
//                auto caption = cv::format("%s", name);
//                int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
//                cv::rectangle(job->getImg(), cv::Point(left - 3, top - 33), cv::Point(left + text_width, top), color, -1);
//                cv::putText(job->getImg(), caption, cv::Point(left, top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
            }
        }


    private:
        const int maxBatchSize_;
        const int maxWorkspace_;
        const int ih_, iw_;
        const int n_, c_, h_, w_, inputSize_;
        int boxNum_, probNum_, classNum_, outputSize_;
        const float confTH_;
        const float nmsTH_;

        AffineMatrix *am_;

        cv::VideoWriter writer_;
        shared_ptr<EngineContext> exeCtx_ = nullptr;
        shared_ptr<JobManager> jobManager_ = nullptr;
        list<shared_ptr<Job>> jobs_;
        list<shared_ptr<Job>> batch_;
        queue<shared_ptr<Job>> results_;

        mutex jobLock_;
        mutex resLock_;
        mutex jobManagerLock_;
        condition_variable jobCV_;
        condition_variable cv_;
        condition_variable resCV_;
        thread workerThread_;
        thread resThread_;
        bool stop_ = false;
    };
}