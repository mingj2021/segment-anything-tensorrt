#ifndef GENERAL_H
#define GENERAL_H

#include <NvInfer.h>
#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>
#include <sstream>

namespace algorithms
{
    uint32_t getElementSize(nvinfer1::DataType t) noexcept
    {
        switch (t)
        {
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kINT8:
            return 1;
        }
        return 0;
    }

    template <class Type>
    Type string2Num(const std::string &str)
    {
        std::istringstream iss(str);
        Type num;
        iss >> std::hex >> num;
        return num;
    }

    std::vector<std::string> read_names(const std::string filename)
    {
        std::vector<std::string> names;
        std::ifstream infile(filename);
        //assert(stream.is_open());

        std::string line;
        while (std::getline(infile, line))
        {
            names.emplace_back(line);
        }
        return names;
    }
}
#endif
