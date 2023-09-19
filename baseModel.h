#ifndef BASEMODEL_H
#define BASEMODEL_H

#include "sam_utils.h"

class BaseModel
{
public:
    cudaStream_t stream;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    std::vector<void *> mDeviceBindings;
    std::map<std::string, std::unique_ptr<algorithms::DeviceBuffer>> mInOut;
    std::vector<std::string> mInputsName, mOutputsName;

public:
    BaseModel(std::string modelFile);
    ~BaseModel();
    void read_engine_file(std::string modelFile);
    // std::vector<std::string> get_inputs_name();
    // std::vector<std::string> get_outputs_name();
    // std::vector<void *> get_device_buffer();
};

BaseModel::BaseModel(std::string modelFile)
{
    read_engine_file(modelFile);
    context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cerr << "create context error" << std::endl;
    }

    CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        auto dims = mEngine->getBindingDimensions(i);
        auto tensor_name = mEngine->getBindingName(i);
        bool isInput = mEngine->bindingIsInput(i);
        if (isInput)
            mInputsName.emplace_back(tensor_name);
        else
            mOutputsName.emplace_back(tensor_name);
        std::cout << "tensor_name: " << tensor_name << std::endl;
        dims2str(dims);
        nvinfer1::DataType type = mEngine->getBindingDataType(i);
        index2srt(type);
        auto vol = std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
        std::unique_ptr<algorithms::DeviceBuffer> device_buffer{new algorithms::DeviceBuffer(vol, type)};
        mDeviceBindings.emplace_back(device_buffer->data());
        mInOut[tensor_name] = std::move(device_buffer);
    }
}

BaseModel::~BaseModel()
{
    CHECK(cudaStreamDestroy(stream));
}

void BaseModel::read_engine_file(std::string modelFile)
{
    std::ifstream engineFile(modelFile.c_str(), std::ifstream::binary);
    assert(engineFile);

    int fsize;
    engineFile.seekg(0, engineFile.end);
    fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    if (engineFile)
        std::cout << "all characters read successfully." << std::endl;
    else
        std::cout << "error: only " << engineFile.gcount() << " could be read" << std::endl;
    engineFile.close();

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
}

#endif