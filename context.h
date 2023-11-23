#include <NvOnnxParser.h>
#include "utils.h"

namespace TRT {
    using namespace nvinfer1;
    using namespace std;

    class EngineContext {
    public:
        virtual ~EngineContext() { destroy(); }

        EngineContext() = default;

        EngineContext(const string& src) {
            auto engineData = loadFile(src);
            if (engineData.empty()) {
                cout << "Read empty file, check path." << endl;
                exit(1);
            }
            runtime_.reset(createInferRuntime(logger));
            engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineData.size()));
            checkNullptr(engine_.get(), "engine");
            exeCtx_ = nvshared(engine_->createExecutionContext());
            checkNullptr(exeCtx_.get(), "exeCtx");
        }

        EngineContext(const string& src, const string& dst, int maxBatchSize, const size_t maxWorkspaceSize) {
            if (!compile(src, dst, maxBatchSize, maxWorkspaceSize)) {
                printf("Compile ONNX file failed, exit.\n");
                return;
            }
            checkNullptr(engine_.get(), "engine");
            exeCtx_ = nvshared(engine_->createExecutionContext());
            checkNullptr(exeCtx_.get(), "exeCtx");
        }


        bool createContext(const void* engineData, size_t size) {
            if (engineData == nullptr || size == 0) return false;
            destroy();

            runtime_ = nvshared(createInferRuntime(logger));
            if (runtime_ == nullptr) return false;
            engine_ = nvshared(runtime_->deserializeCudaEngine(engineData, size));
            if (engine_ == nullptr) return false;
            exeCtx_ = nvshared(engine_->createExecutionContext());
            return exeCtx_ != nullptr;
        }

        bool compile(const string& src, const string& dst, int maxBatchSize, const size_t maxWorkspaceSize) {
            // init
            auto builder = nvshared(nvinfer1::createInferBuilder(logger));
            checkNullptr(builder.get(), "builder");
            auto config = nvshared(builder->createBuilderConfig());
            checkNullptr(config.get(), "config");
            auto explictBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
            auto network = nvshared(builder->createNetworkV2(explictBatch));
            checkNullptr(network.get(), "network");
            auto parser = nvonnxparser::createParser(*network, logger);
            checkNullptr(parser, "parser");
            parser->parseFromFile(src.c_str(), 1);
            builder->setMaxBatchSize(maxBatchSize);
            config->setMaxWorkspaceSize(maxWorkspaceSize);
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            
            // set profile
            auto profile = builder->createOptimizationProfile();
            checkNullptr(profile, "profile");
            int inputSize = network->getNbInputs();
            for (int i = 0; i < inputSize; ++i) {
                auto input = network->getInput(i);
                auto inputDims = input->getDimensions();
                inputDims.d[0] = 1;
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, inputDims);
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, inputDims);
                inputDims.d[1] = maxBatchSize;
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, inputDims);
            }
            config->addOptimizationProfile(profile);

            // generate engine
            engine_.reset(builder->buildEngineWithConfig(*network, *config));
            auto engineData = engine_->serialize();
            return saveFile(dst, engineData->data(), engineData->size());
        }

    private:
        void destroy() {
            exeCtx_.reset();
            engine_.reset();
            runtime_.reset();
        }

    public:
        shared_ptr <IRuntime> runtime_ = nullptr;
        shared_ptr <ICudaEngine> engine_ = nullptr;
        shared_ptr <IExecutionContext> exeCtx_ = nullptr;
    };
}
