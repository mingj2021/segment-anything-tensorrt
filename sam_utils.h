#ifndef SAM_UTILS_H
#define SAM_UTILS_H

#include <NvInfer.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>


using namespace nvinfer1;
using namespace nvonnxparser;

#undef CHECK
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

void index2srt(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        std::cout << "nvinfer1::DataType::kFLOAT" << std::endl;
        break;
    case nvinfer1::DataType::kHALF:
        std::cout << "nvinfer1::DataType::kHALF" << std::endl;
        break;
    case nvinfer1::DataType::kINT8:
        std::cout << "nvinfer1::DataType::kINT8" << std::endl;
        break;
    case nvinfer1::DataType::kINT32:
        std::cout << "nvinfer1::DataType::kINT32" << std::endl;
        break;
    case nvinfer1::DataType::kBOOL:
        std::cout << "nvinfer1::DataType::kBOOL" << std::endl;
        break;
    case nvinfer1::DataType::kUINT8:
        std::cout << "nvinfer1::DataType::kUINT8" << std::endl;
        break;

    default:
        break;
    }
}

void dims2str(nvinfer1::Dims dims)
{
    std::string o_s("[");
    for (size_t i = 0; i < dims.nbDims; i++)
    {
        if (i > 0)
            o_s += ", ";
        o_s += std::to_string(dims.d[i]);
    }
    o_s += "]";
    std::cout << o_s << std::endl;
}
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

#endif