#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "yolo.h"

using namespace TRT;
using namespace YOLO;

int main() {
    cv::VideoCapture cap(R"(/tmp/tmp.kBeZr48XTi/workspace/4k-tokyo-drive-thru-ikebukuro.mp4)");
    float fps = cap.get(cv::CAP_PROP_FPS);
    int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    Size imgSize(height, width);
    int maxBatchSize = 3, c = 3, h = 640, w = 640;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t workspace = 1 << 27;
    Yolo yolo(R"(/tmp/tmp.kBeZr48XTi/workspace/yolov5s.trtmodel)",
              maxBatchSize,
              workspace,
              stream,
              imgSize);
    yolo.startup();

    cv::Mat image, scence;
    scence = cv::Mat(height * 1.5, width * 2, CV_8UC3, cv::Scalar::all(0));
    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('m', 'p', 'g', '2'), fps, scence.size());
    yolo.setWriter(writer);
    auto start = std::chrono::high_resolution_clock::now(); // get current time

    while(cap.read(image)){
        yolo.commit(image);
    }
    auto end = std::chrono::high_resolution_clock::now(); // get current time again
    std::chrono::duration<double, std::milli> duration = end - start; // calculate duration in milliseconds
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl; // print duration
    yolo.stop();
    return 0;
}