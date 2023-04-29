#include "buffers.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
// #include <torchvision/vision.h>
#include <torch/script.h>

using namespace torch::indexing;

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

// class TensorContainer : public torch::nn::Module
// {

// }

class ResizeLongestSide
{
public:
    ResizeLongestSide(int target_length);
    ~ResizeLongestSide();

    at::IntArrayRef get_preprocess_shape(int oldh, int oldw);
    void get_preprocess_shape(int oldh, int oldw,int &newh, int &neww);
    at::Tensor apply_coords(at::Tensor boxes, at::IntArrayRef sz);

public:
    int m_target_length;
};

ResizeLongestSide::ResizeLongestSide(int target_length) : m_target_length(target_length)
{
}

ResizeLongestSide::~ResizeLongestSide()
{
}

at::IntArrayRef ResizeLongestSide::get_preprocess_shape(int oldh, int oldw)
{
    float scale = m_target_length * 1.0 / std::max(oldh, oldw);
    const int newh = static_cast<int>(oldh * scale + 0.5);
    const int neww = static_cast<int>(oldw * scale + 0.5);
    return  at::IntArrayRef{newh, neww};
}

void ResizeLongestSide::get_preprocess_shape(int oldh, int oldw,int &newh, int &neww)
{
    float scale = m_target_length * 1.0 / std::max(oldh, oldw);
    newh = static_cast<int>(oldh * scale + 0.5);
    neww = static_cast<int>(oldw * scale + 0.5);
}

at::Tensor ResizeLongestSide::apply_coords(at::Tensor coords, at::IntArrayRef sz)
{
    int old_h = sz[0], old_w = sz[1];
    int new_h , new_w ;
    get_preprocess_shape(old_h, old_w,new_h,new_w);
    
    coords.index_put_({"...", 0}, coords.index({"...", 0}) * (1.0 * new_w / old_w));
    coords.index_put_({"...", 1}, coords.index({"...", 1}) * (1.0 * new_h / old_h));
    return coords;
}

////////////////////////////////////////////////////////////////////////////////////

class SamEmbedding
{
public:
    SamEmbedding(const std::string &bufferName, std::shared_ptr<nvinfer1::ICudaEngine> &engine, cv::Mat im, int width = 640, int height = 640);
    ~SamEmbedding();

    int prepareInput();
    bool infer();
    int verifyOutput();

public:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream;
    cudaEvent_t start, end;

    std::vector<void *> mDeviceBindings;
    std::map<std::string, std::unique_ptr<algorithms::DeviceBuffer>> mInOut;
    std::vector<float> pad_info;
    std::vector<std::string> names;
    cv::Mat frame;
    cv::Mat img;
    int inp_width = 640;
    int inp_height = 640;
    std::string mBufferName;
};

SamEmbedding::SamEmbedding(const std::string &bufferName, std::shared_ptr<nvinfer1::ICudaEngine> &engine, cv::Mat im, int width, int height) : mBufferName(bufferName), mEngine(engine), frame(im), inp_width(width), inp_height(height)
{
    context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cerr << "create context error" << std::endl;
    }

    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        auto dims = mEngine->getBindingDimensions(i);
        auto tensor_name = mEngine->getBindingName(i);
        std::cout << "tensor_name: " << tensor_name << std::endl;
        dims2str(dims);
        nvinfer1::DataType type = mEngine->getBindingDataType(i);
        index2srt(type);
        int vecDim = mEngine->getBindingVectorizedDim(i);
        // std::cout << "vecDim:" << vecDim << std::endl;
        if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
        {
            int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
            std::cout << "scalarsPerVec" << scalarsPerVec << std::endl;
        }
        auto vol = std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
        std::unique_ptr<algorithms::DeviceBuffer> device_buffer{new algorithms::DeviceBuffer(vol, type)};
        mDeviceBindings.emplace_back(device_buffer->data());
        mInOut[tensor_name] = std::move(device_buffer);
    }
}

SamEmbedding::~SamEmbedding()
{
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    CHECK(cudaStreamDestroy(stream));
}

int SamEmbedding::prepareInput()
{
    int prompt_embed_dim = 256;
    int image_size = 1024;
    int vit_patch_size = 16;
    int target_length = image_size;
    auto pixel_mean = at::tensor({123.675, 116.28, 103.53}, torch::kFloat).view({-1, 1, 1});
    auto pixel_std = at::tensor({58.395, 57.12, 57.375}, torch::kFloat).view({-1, 1, 1});
    ResizeLongestSide transf(image_size);
    int newh,neww;
    transf.get_preprocess_shape(frame.rows, frame.cols,newh,neww);
    cv::Mat im_sz;
    cv::resize(frame, im_sz, cv::Size(neww, newh));
    im_sz.convertTo(im_sz, CV_32F, 1.0);
    at::Tensor input_image_torch =
        at::from_blob(im_sz.data, {im_sz.rows, im_sz.cols, im_sz.channels()})
            .permute({2, 0, 1})
            .contiguous()
            .unsqueeze(0);
    input_image_torch = (input_image_torch - pixel_mean) / pixel_std;
    int h = input_image_torch.size(2);
    int w = input_image_torch.size(3);
    int padh = image_size - h;
    int padw = image_size - w;
    input_image_torch = at::pad(input_image_torch, {0, padw, 0, padh});
    auto ret = mInOut["images"]->host2device((void *)(input_image_torch.data_ptr<float>()), true, stream);
    return ret;
}

bool SamEmbedding::infer()
{
    CHECK(cudaEventRecord(start, stream));
    auto ret = context->enqueueV2(mDeviceBindings.data(), stream, nullptr);
    return ret;
}

int SamEmbedding::verifyOutput()
{
    float ms{0.0f};
    CHECK(cudaEventRecord(end, stream));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&ms, start, end));

    auto dim0 = mEngine->getTensorShape("image_embeddings");

    // dims2str(dim0);
    // dims2str(dim1);
    at::Tensor preds;
    preds = at::zeros({dim0.d[0], dim0.d[1], dim0.d[2], dim0.d[3]}, at::kFloat);
    mInOut["image_embeddings"]->device2host((void *)(preds.data_ptr<float>()), stream);

    // Wait for the work in the stream to complete
    CHECK(cudaStreamSynchronize(stream));
    torch::save({preds}, "preds.pt");
    // cv::FileStorage storage("1.yaml", cv::FileStorage::WRITE);
    // storage << "image_embeddings" << points3dmatrix;
    return 0;
}

///////////////////////////////////////////////////
class SamPromptEncoderAndMaskDecoder
{
public:
    SamPromptEncoderAndMaskDecoder(const std::string &bufferName, std::shared_ptr<nvinfer1::ICudaEngine> &engine, cv::Mat im, int width = 640, int height = 640);
    ~SamPromptEncoderAndMaskDecoder();

    int prepareInput(int x, int y);
    int prepareInput(at::Tensor image_embeddings,at::Tensor point_coords,at::Tensor point_labels,at::Tensor mask_input,at::Tensor has_mask_input);
    bool infer();
    int verifyOutput();
    at::Tensor generator_colors(int num);

    template <class Type>
    Type string2Num(const std::string &str);

    at::Tensor plot_masks(at::Tensor masks, at::Tensor im_gpu, float alpha);

public:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream;
    cudaEvent_t start, end;

    std::vector<void *> mDeviceBindings;
    std::map<std::string, std::unique_ptr<algorithms::DeviceBuffer>> mInOut;
    std::vector<float> pad_info;
    std::vector<std::string> names;
    cv::Mat frame;
    cv::Mat img;
    int inp_width = 640;
    int inp_height = 640;
    std::string mBufferName;
};

SamPromptEncoderAndMaskDecoder::SamPromptEncoderAndMaskDecoder(const std::string &bufferName, std::shared_ptr<nvinfer1::ICudaEngine> &engine, cv::Mat im, int width, int height) : mBufferName(bufferName), mEngine(engine), frame(im), inp_width(width), inp_height(height)
{
    context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cerr << "create context error" << std::endl;
    }
    // set input dims whichs name "point_coords "
    context->setBindingDimensions(1, nvinfer1::Dims3(1, 2, 2));
    // set input dims whichs name "point_label "
    context->setBindingDimensions(2, nvinfer1::Dims2(1, 2));
    // set input dims whichs name "point_label "
    // context->setBindingDimensions(5, nvinfer1::Dims2(frame.rows,frame.cols));
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    int nbopts = mEngine->getNbOptimizationProfiles();
    std::cout << "nboopts: " << nbopts << std::endl;
    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        // auto dims = mEngine->getBindingDimensions(i);
        auto tensor_name = mEngine->getBindingName(i);
        std::cout << "tensor_name: " << tensor_name << std::endl;
        auto dims = context->getBindingDimensions(i);
        dims2str(dims);
        nvinfer1::DataType type = mEngine->getBindingDataType(i);
        index2srt(type);
        auto vol = std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
        std::unique_ptr<algorithms::DeviceBuffer> device_buffer{new algorithms::DeviceBuffer(vol, type)};
        mDeviceBindings.emplace_back(device_buffer->data());
        mInOut[tensor_name] = std::move(device_buffer);
    }
}

SamPromptEncoderAndMaskDecoder::~SamPromptEncoderAndMaskDecoder()
{
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    CHECK(cudaStreamDestroy(stream));
}

int SamPromptEncoderAndMaskDecoder::prepareInput(int x, int y)
{
    at::Tensor image_embeddings;

    torch::load(image_embeddings, "preds.pt");
    // std::cout << image_embeddings.sizes() << std::endl;
    int image_size = 1024;
    ResizeLongestSide transf(image_size);

    auto input_point = at::tensor({x, y}, at::kFloat).reshape({-1,2});
    auto input_label = at::tensor({1}, at::kFloat);

    auto trt_coord = at::concatenate({input_point, at::tensor({0, 0}, at::kFloat).unsqueeze(0)}, 0).unsqueeze(0);
    auto trt_label = at::concatenate({input_label, at::tensor({-1}, at::kFloat)}, 0).unsqueeze(0);
    trt_coord = transf.apply_coords(trt_coord, {frame.rows, frame.cols});

    auto trt_mask_input = at::zeros({1, 1, 256, 256}, at::kFloat);
    auto trt_has_mask_input = at::zeros(1, at::kFloat);

    CHECK(mInOut["image_embeddings"]->host2device((void *)(image_embeddings.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(mInOut["point_coords"]->host2device((void *)(trt_coord.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(mInOut["point_labels"]->host2device((void *)(trt_label.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(mInOut["mask_input"]->host2device((void *)(trt_mask_input.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(mInOut["has_mask_input"]->host2device((void *)(trt_has_mask_input.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    return 0;
}

int SamPromptEncoderAndMaskDecoder::prepareInput(at::Tensor image_embeddings,at::Tensor point_coords,at::Tensor point_labels,at::Tensor mask_input,at::Tensor has_mask_input)
{
    context.reset(mEngine->createExecutionContext());
    // set input dims whichs name "point_coords "
    context->setBindingDimensions(1, nvinfer1::Dims3(point_coords.size(0), point_coords.size(1), point_coords.size(2)));
    // set input dims whichs name "point_label "
    context->setBindingDimensions(2, nvinfer1::Dims2(point_coords.size(0), point_coords.size(1)));
    return 0;

}

bool SamPromptEncoderAndMaskDecoder::infer()
{
    CHECK(cudaEventRecord(start, stream));
    auto ret = context->enqueueV2(mDeviceBindings.data(), stream, nullptr);
    return ret;
}

int SamPromptEncoderAndMaskDecoder::verifyOutput()
{
    float ms{0.0f};
    CHECK(cudaEventRecord(end, stream));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&ms, start, end));

    auto dim0 = mEngine->getTensorShape("masks");
    auto dim1 = mEngine->getTensorShape("iou_predictions");
    // dims2str(dim0);
    // dims2str(dim1);
    at::Tensor masks;
    masks = at::zeros({dim0.d[0], dim0.d[1], dim0.d[2], dim0.d[3]}, at::kFloat);
    mInOut["masks"]->device2host((void *)(masks.data_ptr<float>()), stream);
    // Wait for the work in the stream to complete
    CHECK(cudaStreamSynchronize(stream));

    int longest_side = 1024;

    namespace F = torch::nn::functional;
    masks = F::interpolate(masks, F::InterpolateFuncOptions().size(std::vector<int64_t>({longest_side, longest_side})).mode(torch::kBilinear).align_corners(false));
    // at::IntArrayRef input_image_size{frame.rows, frame.cols};
    ResizeLongestSide transf(longest_side);
    int newh,neww;
    transf.get_preprocess_shape(frame.rows, frame.cols,newh,neww);
    masks = masks.index({"...", Slice(None, newh), Slice(None, neww)});

    masks = F::interpolate(masks, F::InterpolateFuncOptions().size(std::vector<int64_t>({frame.rows, frame.cols})).mode(torch::kBilinear).align_corners(false));
    std::cout << "masks: " << masks.sizes() << std::endl;

    at::Tensor iou_predictions;
    iou_predictions = at::zeros({dim0.d[0], dim0.d[1]}, at::kFloat);
    mInOut["scores"]->device2host((void *)(iou_predictions.data_ptr<float>()), stream);
    // Wait for the work in the stream to complete
    CHECK(cudaStreamSynchronize(stream));

    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    torch::Device device(device_type);
    masks = masks.gt(0.) * 1.0;
    std::cout << "max " << masks.max() << std::endl;
    // masks = masks.sigmoid();
    std::cout << "masks: " << masks.sizes() << std::endl;
    masks = masks.to(device);
    std::cout << "iou_predictions: " << iou_predictions << std::endl;
    cv::Mat img;
    // cv::Mat frame = cv::imread("D:/projects/detections/data/truck.jpg");
    frame.convertTo(img, CV_32F, 1.0 / 255);
    at::Tensor im_gpu =
        at::from_blob(img.data, {img.rows, img.cols, img.channels()})
            .permute({2, 0, 1})
            .contiguous()
            .to(device);
    auto results = plot_masks(masks, im_gpu, 0.5);
    auto t_img = results.to(torch::kCPU).clamp(0, 255).to(torch::kU8);

    auto img_ = cv::Mat(t_img.size(0), t_img.size(1), CV_8UC3, t_img.data_ptr<uchar>());
    cv::cvtColor(img_, img_, cv::COLOR_RGB2BGR);
    // cv::namedWindow("img_", 0);
    cv::imshow("img_", img_);
    // cv::waitKey();
    return 0;
}

/*
    return [r g b] * n
*/
at::Tensor SamPromptEncoderAndMaskDecoder::generator_colors(int num)
{

    std::vector<std::string> hexs = {"FF37C7", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
                                     "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "", "520085", "CB38FF", "FF95C8", "FF3838"};

    std::vector<int> tmp;
    for (int i = 0; i < num; ++i)
    {
        int r = string2Num<int>(hexs[i].substr(0, 2));
        // std::cout << r << std::endl;
        int g = string2Num<int>(hexs[i].substr(2, 2));
        // std::cout << g << std::endl;
        int b = string2Num<int>(hexs[i].substr(4, 2));
        // std::cout << b << std::endl;
        tmp.emplace_back(r);
        tmp.emplace_back(g);
        tmp.emplace_back(b);
    }
    return at::from_blob(tmp.data(), {(int)tmp.size()}, at::TensorOptions(at::kInt));
}

template <class Type>
Type SamPromptEncoderAndMaskDecoder::string2Num(const std::string &str)
{
    std::istringstream iss(str);
    Type num;
    iss >> std::hex >> num;
    return num;
}

/*
        Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
*/

at::Tensor SamPromptEncoderAndMaskDecoder::plot_masks(at::Tensor masks, at::Tensor im_gpu, float alpha)
{
    int n = masks.size(0);
    auto colors = generator_colors(n);
    colors = colors.to(masks.device()).to(at::kFloat).div(255).reshape({-1, 3}).unsqueeze(1).unsqueeze(2);
    // std::cout << "colors: " << colors.sizes() << std::endl;
    masks = masks.permute({0, 2, 3, 1}).contiguous();
    // std::cout << "masks: " << masks.sizes() << std::endl;
    auto masks_color = masks * (colors * alpha);
    // std::cout << "masks_color: " << masks_color.sizes() << std::endl;
    auto inv_alph_masks = (1 - masks * alpha);
    inv_alph_masks = inv_alph_masks.cumprod(0);
    // std::cout << "inv_alph_masks: " << inv_alph_masks.sizes() << std::endl;

    auto mcs = masks_color * inv_alph_masks;
    mcs = mcs.sum(0) * 2;
    // std::cout << "mcs: " << mcs.sizes() << std::endl;
    im_gpu = im_gpu.flip({0});
    // std::cout << "im_gpu: " << im_gpu.sizes() << std::endl;
    im_gpu = im_gpu.permute({1, 2, 0}).contiguous();
    // std::cout << "im_gpu: " << im_gpu.sizes() << std::endl;
    im_gpu = im_gpu * inv_alph_masks[-1] + mcs;
    // std::cout << "im_gpu: " << im_gpu.sizes() << std::endl;
    auto im_mask = (im_gpu * 255);
    return im_mask;
}
///////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;

std::shared_ptr<SamPromptEncoderAndMaskDecoder> b;

void locator(int event, int x, int y, int flags, void *userdata)
{ // function to track mouse movement and click//
    if (event == EVENT_LBUTTONDOWN)
    { // when left button clicked//
        //   cout << "Left click has been made, Position:(" << x << "," << y << ")" << endl;
        auto res = b->prepareInput(x, y);
        std::cout << "------------------prepareInput: " << res << std::endl;
        res = b->infer();
        std::cout << "------------------infer: " << res << std::endl;
        b->verifyOutput();
        std::cout << "------------------verifyOutput: " << std::endl;
    }
    else if (event == EVENT_RBUTTONDOWN)
    { // when right button clicked//
        //   cout << "Rightclick has been made, Position:(" << x << "," << y << ")" << endl;
    }
    else if (event == EVENT_MBUTTONDOWN)
    { // when middle button clicked//
        //   cout << "Middleclick has been made, Position:(" << x << "," << y << ")" << endl;
    }
    else if (event == EVENT_MOUSEMOVE)
    { // when mouse pointer moves//
        //   cout << "Current mouse position:(" << x << "," << y << ")" << endl;
    }
}

// #define EMBEDDING
#define SAMPROMPTENCODERANDMASKDECODER
int main(int argc, char const *argv[])
{
    std::cout << at::IntArrayRef{1,2} << std::endl;

#ifdef EMBEDDING
    const std::string modelFile = "/workspace/segment-anything-tensorrt/segment-anything/vit_l_embedding.engine";
    std::ifstream engineFile(modelFile.c_str(), std::ifstream::binary);
    assert(engineFile);
    // if (!engineFile)
    //     return;

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
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    cv::Mat frame = cv::imread("/workspace/segment-anything-tensorrt/segment-anything/notebooks/images/truck.jpg");
    std::cout << frame.size() << std::endl;
    std::shared_ptr<SamEmbedding> b(new SamEmbedding(std::to_string(1), mEngine, frame));
    auto res = b->prepareInput();
    std::cout << "------------------prepareInput: " << res << std::endl;
    res = b->infer();
    std::cout << "------------------infer: " << res << std::endl;
    b->verifyOutput();
    std::cout << "------------------verifyOutput: " << std::endl;
#endif

#ifdef SAMPROMPTENCODERANDMASKDECODER
    const std::string modelFile = "/workspace/segment-anything-tensorrt/segment-anything/sam_onnx_example.engine";
    std::ifstream engineFile(modelFile.c_str(), std::ifstream::binary);
    assert(engineFile);
    // if (!engineFile)
    //     return;

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
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    cv::Mat frame = cv::imread("/workspace/segment-anything-tensorrt/segment-anything/notebooks/images/truck.jpg");
    b = std::shared_ptr<SamPromptEncoderAndMaskDecoder>(new SamPromptEncoderAndMaskDecoder(std::to_string(1), mEngine, frame));
    //  Mat image = imread("D:/projects/detections/data/2.png");//loading image in the matrix//
    namedWindow("img_", 0);                  // declaring window to show image//
    setMouseCallback("img_", locator, NULL); // Mouse callback function on define window//
    imshow("img_", frame);                   // showing image on the window//
    waitKey(0);                              // wait for keystroke//

#endif
}