import torch
import torch.nn as nn
from torch.nn import functional as F
from segment_anything.modeling import Sam
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
from typing import Tuple
from segment_anything import sam_model_registry, SamPredictor
import cv2
import matplotlib.pyplot as plt
import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import onnxruntime

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

# @torch.no_grad()
def pre_processing(image: np.ndarray, target_length: int, device,pixel_mean,pixel_std,img_size):
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)
    input_image = cv2.resize(image,target_size)
    # input_image = np.array(resize(to_pil_image(image), target_size))
    input_image_torch = torch.as_tensor(input_image, device=device)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    # Normalize colors
    input_image_torch = (input_image_torch - pixel_mean) / pixel_std

    # Pad
    h, w = input_image_torch.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
    return input_image_torch

# class EmbeddingOnnxModel(nn.Module):
#     def __init__(self, sam_model: Sam):
#         super().__init__()
#         self.image_encoder = sam_model.image_encoder

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         return self.image_encoder(x)
    
def test_predictor():
    sam_checkpoint = "weights/sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    image = cv2.imread('notebooks/images/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = image.copy()

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()

    predictor.set_image(image)
    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show() 
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    

    target_length = sam.image_encoder.img_size
    pixel_mean = sam.pixel_mean 
    pixel_std = sam.pixel_std
    img_size = sam.image_encoder.img_size
    # inputs1 = predictor.input_image
    inputs2 = pre_processing(image2, target_length, device,pixel_mean,pixel_std,img_size)
    image_embedding2 = onnx_model_example()
    print(image_embedding.shape)
    print(image_embedding2.shape)
    print(np.array_equal(image_embedding ,image_embedding2))

def export_embedding_model():
    sam_checkpoint = "weights/sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # image_encoder = EmbeddingOnnxModel(sam)
    image = cv2.imread('notebooks/images/truck.jpg')
    target_length = sam.image_encoder.img_size
    pixel_mean = sam.pixel_mean 
    pixel_std = sam.pixel_std
    img_size = sam.image_encoder.img_size
    inputs = pre_processing(image, target_length, device,pixel_mean,pixel_std,img_size)
    onnx_model_path = model_type+"_"+"embedding.onnx"
    dummy_inputs = {
    "images": inputs
}
    output_names = ["image_embeddings"]
    image_embeddings = sam.image_encoder(inputs).cpu().numpy()
    print('image_embeddings', image_embeddings.shape)
    # with warnings.catch_warnings():
        # warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        # warnings.filterwarnings("ignore", category=UserWarning)
    with open(onnx_model_path, "wb") as f:
        torch.onnx.export(
            sam.image_encoder,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=12,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            # dynamic_axes=dynamic_axes,
        )    

def test_save_parameters():
    # from functools import partial
    # from copy import deepcopy
    sam_checkpoint = "weights/sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # encoder_embed_dim=1024,
    # encoder_depth=24,
    # encoder_num_heads=16,
    # encoder_global_attn_indexes=[5, 11, 17, 23]
    # prompt_embed_dim = 256
    # image_size = 1024
    # vit_patch_size = 16
    # image_embedding_size = image_size // vit_patch_size
    # model = ImageEncoderViT(
    #         depth=encoder_depth,
    #         embed_dim=encoder_embed_dim,
    #         img_size=image_size,
    #         mlp_ratio=4,
    #         norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    #         num_heads=encoder_num_heads,
    #         patch_size=vit_patch_size,
    #         qkv_bias=True,
    #         use_rel_pos=True,
    #         global_attn_indexes=encoder_global_attn_indexes,
    #         window_size=14,
    #         out_chans=prompt_embed_dim,
    #     )
    # model.to(device=device)
    # b = torch.load(sam_checkpoint)
    # a = sam.state_dict()
    # model.load_state_dict(sam.image_encoder.state_dict()) # copy weights and stuff

    # for name, param in model.named_parameters():
    #     print('name: ', name)
    #     print(type(param))
    #     print('param.shape: ', param.shape)
    #     print('param.requires_grad: ', param.requires_grad)
    #     print('=====')
    
    # torch.save(model, 'image_encoder.pth')

def onnx_model_example():
    model_type = "vit_l"
    onnx_model_path = model_type+"_"+"embedding.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path,providers=['CPUExecutionProvider'])
    image = cv2.imread('notebooks/images/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    target_length = image_size
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375]
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    device = "cpu"
    inputs = pre_processing(image, target_length, device,pixel_mean,pixel_std,image_size)
    ort_inputs = {
    "images": inputs.cpu().numpy()
    }
    image_embeddings = ort_session.run(None, ort_inputs)
    print(image_embeddings[0].shape)
    return image_embeddings[0]

def export_sam_model():
    from segment_anything.utils.onnx import SamOnnxModel
    checkpoint = "weights/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    onnx_model_path = "sam_onnx_example.onnx"

    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float)
        # "orig_im_size": torch.tensor([1500, 2250], dtype=torch.int32),
    }
    # output_names = ["masks", "iou_predictions", "low_res_masks"]
    output_names = ["masks", "iou_predictions"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=12,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )  

def main():
    import os
    ort_session_embedding = onnxruntime.InferenceSession('vit_l_embedding.onnx',providers=['CPUExecutionProvider'])
    ort_session_sam = onnxruntime.InferenceSession('sam_onnx_example.onnx',providers=['CPUExecutionProvider'])

    image = cv2.imread('notebooks/images/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = image.copy()
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    target_length = image_size
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375]
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    device = "cpu"
    inputs = pre_processing(image, target_length, device,pixel_mean,pixel_std,image_size)
    ort_inputs = {
    "images": inputs.cpu().numpy()
    }
    if not os.path.exists('image_embeddings.npz'):
        image_embeddings = ort_session_embedding.run(None, ort_inputs)[0]
        np.savez('image_embeddings.npz',image_embeddings=image_embeddings)
    else:
        image_embeddings = np.load('image_embeddings.npz')['image_embeddings']

    # image_embeddings = torch.load('preds.pt')
    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    from segment_anything.utils.transforms import ResizeLongestSide
    transf = ResizeLongestSide(image_size)
    onnx_coord = transf.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    # print(onnx_coord)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    # checkpoint = "weights/sam_vit_l_0b3195.pth"
    # model_type = "vit_l"
    # sam = sam_model_registry[model_type](checkpoint=checkpoint)
    # sam.to(device='cpu')
    # predictor = SamPredictor(sam)
    # predictor.set_image(image2)
    # image_embedding = predictor.get_image_embedding().cpu().numpy()

    ort_inputs = {
        "image_embeddings": image_embeddings,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input
        # "orig_im_size": np.array(image.shape[:2], dtype=np.int32)
    }

    masks, scores = ort_session_sam.run(None, ort_inputs)
    from segment_anything.utils.onnx import SamOnnxModel
    checkpoint = "weights/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    onnx_model_path = "sam_onnx_example.onnx"

    onnx_model = SamOnnxModel(sam, return_single_mask=True)
    # masks = onnx_model.mask_postprocessing(torch.as_tensor(masks), torch.as_tensor(image.shape[:2])).squeeze(0).permute(1,2,0).numpy()
    masks = onnx_model.mask_postprocessing(torch.as_tensor(masks), torch.as_tensor(image.shape[:2]))
    masks = masks > 0.0
    # img = masks * 255
    # img = torch.as_tensor(img).squeeze(0).permute(1,2,0)
    # img = np.asarray(img,np.uint8)
    # cv2.namedWindow('img',0)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    # show_box(input_box, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()

def export_engine(f):
    import tensorrt as trt
    from pathlib import Path
    file = Path(f)
    f = file.with_suffix('.engine')  # TensorRT engine file
    onnx = file.with_suffix('.onnx')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = 10
    print("workspace: ", workspace)
    config.max_workspace_size = workspace * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

    profile = builder.create_optimization_profile()
    profile.set_shape('image_embeddings', (1, 256, 64, 64), (1, 256, 64, 64), (1, 256, 64, 64))
    profile.set_shape('point_coords', (1, 2,2), (1, 5,2), (1,10,2))
    profile.set_shape('point_labels', (1, 2), (1, 5), (1,10))
    profile.set_shape('mask_input', (1, 1, 256, 256), (1, 1, 256, 256), (1, 1, 256, 256))
    profile.set_shape('has_mask_input', (1,), (1, ), (1, ))
    # profile.set_shape_input('orig_im_size', (416,416), (1024,1024), (1500, 2250))
    profile.set_shape_input('orig_im_size', (2,), (2,), (2, ))
    config.add_optimization_profile(profile)

    half = False
    print(f'building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())


# def trt_main():
#     import pycuda.autoinit
#     import tensorrt as trt
#     import sys
#     import os
class TensorContainer(nn.Module):
    def __init__(self, tensor_dict):
        super().__init__()
        for key,value in tensor_dict.items():
            setattr(self, key, value)

if __name__ == '__main__':
    # image_embeddings = np.load('image_embeddings.npz')['image_embeddings']
    # image_embeddings = torch.as_tensor(image_embeddings)
    # tensor_dict = {'image_embeddings': image_embeddings}
    # tensors = TensorContainer(tensor_dict)
    # tensors = torch.jit.script(tensors)
    # tensors.save('image_embeddings.pth')
    # print(np.zeros((1,2)))
    # print(np.zeros((2,)))
    # test_predictor()
    # test_save_parameters()
    with torch.no_grad():
        
        export_embedding_model()
        # export_sam_model()
        # main()
        # test_predictor()
        # export_engine('vit_l_embedding.onnx')
        # export_engine('sam_onnx_example.onnx')