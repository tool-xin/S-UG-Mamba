import os
import sys
import glob
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm

try:
    from models.models import SSUGMamba
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure the 'models' directory (or compiled .so libraries) is in the same directory as this script.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for SeGMaNet (SSUGMamba)")
    
    # 核心路径参数
    parser.add_argument('--img_dir', type=str, required=True, help="Path to the input images directory")
    parser.add_argument('--weight_path', type=str, required=True, help="Path to the model weight (.pt) file")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Directory to save the predicted maps")
    
    # 模型配置参数
    parser.add_argument('--backbone', type=str, default='convnext_tiny', help="Backbone network type")
    parser.add_argument('--disable_mamba', action='store_true', help="Disable Mamba enhancement modules if set")
    parser.add_argument('--img_size', type=int, default=320, help="Input image size (default: 320)")
    
    return parser.parse_args()

def read_image(p, transform):
    img = Image.open(p).convert('RGB')
    original_size = img.size  # (W, H)
    img = transform(img)
    return img[None, :, :, :], original_size

def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    
    img_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(args.img_dir, ext)))
        paths.extend(glob.glob(os.path.join(args.img_dir, ext.upper())))
    
    paths = sorted(list(set(paths)))
    if len(paths) == 0:
        print(f"Error: No images found in {args.img_dir}")
        return

    use_mamba = not args.disable_mamba
    print(f"🚀 Initializing Model | Backbone: {args.backbone} | Mamba Enabled: {use_mamba}")
    
    try:
        model = SSUGMamba(backbone_type=args.backbone, use_mamba=use_mamba)
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return

    print(f"📥 Loading weights from: {args.weight_path}")
    if not os.path.exists(args.weight_path):
        print(f"Error: Weight file does not exist!")
        return

    state_dict = torch.load(args.weight_path, map_location='cpu')
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print(f"Weights loaded successfully.")
    except RuntimeError as e:
        print(f"Weight loading error: \n{e}")
        print("Hint: Please ensure the weight file matches the specified model architecture.")
        return
    
    model.to(device)
    model.eval()

    print(f"📷 Processing {len(paths)} images -> Output directory: {args.output_dir}")
    
    for p in tqdm(paths):
        img, img_size = read_image(p, img_transform)
        filename = os.path.splitext(os.path.basename(p))[0]
        
        with torch.no_grad():
            img = img.to(device)
            output = model(img)
            pred = output[0] if isinstance(output, tuple) else output
            pred_map = pred[0].detach().cpu().numpy()
            pred_map = cv2.resize(pred_map, img_size)
            min_val, max_val = pred_map.min(), pred_map.max()
            pred_map = (pred_map - min_val) / (max_val - min_val + 1e-8)
            #高斯平滑 (Gaussian Blur): 缓解上采样网格伪影，使分布更平滑自然(可选，取决于数据集和竞赛)
            pred_map = cv2.GaussianBlur(pred_map, (7, 7), 2.0)
            pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
            #底值抬升 (Anti-Dead-Zero): 将取值映射至 [1, 255]，防止因纯黑背景导致 KL Divergence 计算时分母为 0(可选，取决于数据集和竞赛)
            pred_map = pred_map * 254 + 1 
            pred_map = np.clip(np.round(pred_map), 1, 255).astype(np.uint8)
            save_path = os.path.join(args.output_dir, filename + ".png")
            cv2.imwrite(save_path, pred_map)

    print(f"Inference Complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()