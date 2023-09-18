# Overview
[中文-windows](README_zh_windows.md)
<p float="left">
  <img src="truck.gif?raw=true" width="100%" />
</p>

download [translated onnx model](https://drive.google.com/drive/folders/1xZ7XpyKGx0Fg-t81SLND56vC2TEQ4otN?usp=sharing)
# tutorials
tutorials.ipynb 

# how to export vim_h model
how_to_export_vim_h_model.ipynb
```
# Divide the embeding model into 2 parts, named as part1 and part2 
# The decoder model remains unchanged
# export_embedding_model_part_1()
# export_embedding_model_part_2()
# export_sam_h_model()
# onnx_model_example2()
```
# FP32 or FP16
```
you can not set FP16 value in export.h, if gpu have enough mems.
```