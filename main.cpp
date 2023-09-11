#include "sam.h"
#include "export.h"
///////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;

std::shared_ptr<SamEmbedding> eng_0;
std::shared_ptr<SamPromptEncoderAndMaskDecoder> eng_1;
at::Tensor image_embeddings;

void locator(int event, int x, int y, int flags, void *userdata)
{ // function to track mouse movement and click//
    if (event == EVENT_LBUTTONDOWN)
    { // when left button clicked//
        cout << "Left click has been made, Position:(" << x << "," << y << ")" << endl;
        // auto res = eng_1->prepareInput(x, y, x - 100, y - 100, x + 100, y + 100, image_embeddings);
        auto res = eng_1->prepareInput(x, y, image_embeddings);
        // std::vector<int> mult_pts = {x,y,x-5,y-5,x+5,y+5};
        // auto res = eng_1->prepareInput(mult_pts, image_embeddings);
        std::cout << "------------------prepareInput: " << res << std::endl;
        res = eng_1->infer();
        std::cout << "------------------infer: " << res << std::endl;
        eng_1->verifyOutput();
        // cv::Mat roi;
        // eng_1->verifyOutput(roi);
        // roi *= 255;
        // cv::imshow("img2_", roi);
        // cv::waitKey();
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

#define EMBEDDING
#define SAMPROMPTENCODERANDMASKDECODER
int main(int argc, char const *argv[])
{
    ifstream f1("vit_l_embedding.engine");
    if (!f1.good())
        export_engine_image_encoder("/workspace/segment-anything-tensorrt/data/vit_l_embedding.onnx");

    ifstream f2("sam_onnx_example.engine");
    if (!f2.good())
        export_engine_prompt_encoder_and_mask_decoder("/workspace/segment-anything-tensorrt/data/sam_onnx_example.onnx");

#ifdef EMBEDDING
    {
        // const std::string modelFile = "D:/projects/detections/data/vit_l_embedding.engine";
        const std::string modelFile = "vit_l_embedding.engine";
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
        cv::Mat frame = cv::imread("/workspace/segment-anything-tensorrt/data/truck.jpg");
        std::cout << frame.size << std::endl;
        eng_0 = std::shared_ptr<SamEmbedding>(new SamEmbedding(std::to_string(1), mEngine, frame));
        auto res = eng_0->prepareInput();
        std::cout << "------------------prepareInput: " << res << std::endl;
        res = eng_0->infer();
        std::cout << "------------------infer: " << res << std::endl;
        image_embeddings = eng_0->verifyOutput();
        std::cout << "------------------verifyOutput: " << std::endl;
    }

#endif

#ifdef SAMPROMPTENCODERANDMASKDECODER
    {
        // const std::string modelFile = "D:/projects/detections/data/sam_onnx_example.engine";
        const std::string modelFile = "sam_onnx_example.engine";
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
        cv::Mat frame = cv::imread("/workspace/segment-anything-tensorrt/data/truck.jpg");
        eng_1 = std::shared_ptr<SamPromptEncoderAndMaskDecoder>(new SamPromptEncoderAndMaskDecoder(std::to_string(1), mEngine, frame));
        //  Mat image = imread("D:/projects/detections/data/2.png");//loading image in the matrix//
        // namedWindow("img_", 0);                  // declaring window to show image//
        // setMouseCallback("img_", locator, NULL); // Mouse callback function on define window//
        // imshow("img_", frame);                   // showing image on the window//
        // waitKey(0);                              // wait for keystroke//

        auto res = eng_1->prepareInput(100, 100, image_embeddings);
        // std::vector<int> mult_pts = {x,y,x-5,y-5,x+5,y+5};
        // auto res = eng_1->prepareInput(mult_pts, image_embeddings);
        std::cout << "------------------prepareInput: " << res << std::endl;
        res = eng_1->infer();
        std::cout << "------------------infer: " << res << std::endl;
        eng_1->verifyOutput();
        std::cout << "-----------------done" << std::endl;
    }
#endif
}