#ifndef EXPORT_H
#define EXPORT_H

#include <iostream>
#include <fstream>
#include "sam_utils.h"


void export_engine_image_encoder(std::string f="vit_l_embedding.onnx")
{
    // create an instance of the builder
    std::unique_ptr<nvinfer1::IBuilder> builder(createInferBuilder(logger));
    // create a network definition
    // The kEXPLICIT_BATCH flag is required in order to import models using the ONNX parser.
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    //auto network = std::make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flag));

    // Importing a Model Using the ONNX Parser
    //auto parser = std::make_unique<nvonnxparser::IParser>(createParser(*network, logger));
    std::unique_ptr<nvonnxparser::IParser> parser(createParser(*network, logger));

    // read the model file and process any errors
    parser->parseFromFile(f.c_str(),
                          static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // create a build configuration specifying how TensorRT should optimize the model
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

    // maximum workspace size
    // int workspace = 4;  // GB
    // config->setMaxWorkspaceSize(workspace * 1U << 30);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    config->setFlag(BuilderFlag::kFP16);

    // create an engine
    // auto serializedModel = std::make_unique<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
     std::unique_ptr<nvinfer1::IHostMemory> serializedModel(builder->buildSerializedNetwork(*network, *config));
     std::cout << "serializedModel->size()" << serializedModel->size() << std::endl;
     std::ofstream outfile("vit_l_embedding.engine", std::ofstream::out | std::ofstream::binary);
     outfile.write((char*)serializedModel->data(), serializedModel->size());
}

void export_engine_prompt_encoder_and_mask_decoder(std::string f="sam_onnx_example.onnx")
{
    // create an instance of the builder
    std::unique_ptr<nvinfer1::IBuilder> builder(createInferBuilder(logger));
    // create a network definition
    // The kEXPLICIT_BATCH flag is required in order to import models using the ONNX parser.
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    //auto network = std::make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flag));

    // Importing a Model Using the ONNX Parser
    //auto parser = std::make_unique<nvonnxparser::IParser>(createParser(*network, logger));
    std::unique_ptr<nvonnxparser::IParser> parser(createParser(*network, logger));

    // read the model file and process any errors
    parser->parseFromFile(f.c_str(),
                          static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // create a build configuration specifying how TensorRT should optimize the model
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

    // maximum workspace size
    // int workspace = 8;  // GB
    // config->setMaxWorkspaceSize(workspace * 1U << 30);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    config->setFlag(BuilderFlag::kFP16);

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    // profile->setDimensions("image_embeddings", nvinfer1::OptProfileSelector::kMIN, {1, 256, 64, 64 });
    // profile->setDimensions("image_embeddings", nvinfer1::OptProfileSelector::kOPT, {1, 256, 64, 64 });
    // profile->setDimensions("image_embeddings", nvinfer1::OptProfileSelector::kMAX, {1, 256, 64, 64 });

    profile->setDimensions("point_coords", nvinfer1::OptProfileSelector::kMIN, {3, 1, 2,2 });
    profile->setDimensions("point_coords", nvinfer1::OptProfileSelector::kOPT, { 3,1, 5,2 });
    profile->setDimensions("point_coords", nvinfer1::OptProfileSelector::kMAX, { 3,1,10,2 });

    profile->setDimensions("point_labels", nvinfer1::OptProfileSelector::kMIN, { 2,1, 2});
    profile->setDimensions("point_labels", nvinfer1::OptProfileSelector::kOPT, { 2,1, 5 });
    profile->setDimensions("point_labels", nvinfer1::OptProfileSelector::kMAX, { 2,1,10 });

    // profile->setDimensions("mask_input", nvinfer1::OptProfileSelector::kMIN, { 1, 1, 256, 256});
    // profile->setDimensions("mask_input", nvinfer1::OptProfileSelector::kOPT, { 1, 1, 256, 256 });
    // profile->setDimensions("mask_input", nvinfer1::OptProfileSelector::kMAX, { 1, 1, 256, 256 });

    // profile->setDimensions("has_mask_input", nvinfer1::OptProfileSelector::kMIN, { 1,});
    // profile->setDimensions("has_mask_input", nvinfer1::OptProfileSelector::kOPT, { 1, });
    // profile->setDimensions("has_mask_input", nvinfer1::OptProfileSelector::kMAX, { 1, });

    config->addOptimizationProfile(profile);

    // create an engine
    // auto serializedModel = std::make_unique<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
     std::unique_ptr<nvinfer1::IHostMemory> serializedModel(builder->buildSerializedNetwork(*network, *config));
     std::cout << "serializedModel->size()" << serializedModel->size() << std::endl;
     std::ofstream outfile("sam_onnx_example.engine", std::ofstream::out | std::ofstream::binary);
     outfile.write((char*)serializedModel->data(), serializedModel->size());
}
#endif