# Overview
[english](README.md)

# cudatoolkit 安装
运行 cuda_11.7.0_516.01_windows.exe ，一路默认即可

# cudnn 安装
解压文件cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip，将解压后的文件夹[lib bin, include] 复制到cuda 对应安装路径[C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7]

# libtorch 安装
下载[https://download.pytorch.org/libtorch/cu117/libtorch-win-shared-with-deps-2.0.1%2Bcu117.zip]并解压。

# torchvision 安装
```
git clone https://github.com/pytorch/vision.git
mkdir build
cd build
cmake .. -DWITH_CUDA=on
cmake --build . --config release --target install
```

# tensorrt 安装
下载[TensorRT-8.5.1.7.Windows10.x86_64.cuda-x.x.zip]并解压

# opencv 安装
下载[https://github.com/opencv/opencv/releases/download/4.7.0/opencv-4.7.0-windows.exe]并默认安装.

