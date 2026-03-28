import time
import dlib
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from dataset import reconstruct_color
from pix2pix_gan import ResnetGenerator

def run_inference():
    parser = argparse.ArgumentParser(description="Start a PyTorch selfie2anime tester.")
    parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
    parser.add_argument("--resize", help="set the short size of fake image (default: 256)", type=int, default=256)
    parser.add_argument("--device", help="select device (cuda, mps, cpu). Auto-detects if not set", type=str, default=None)
    parser.add_argument("--model_path", help="path to the .pth model file", type=str, default="model/selfie2anime.gen_ab.pth")
    args = parser.parse_args()

    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA (NVIDIA GPU)")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("Using CPU")

    print("Loading models...", flush=True)
    det = dlib.get_frontal_face_detector()
    gen = ResnetGenerator().to(device)
    
    # Load parameters (expecting PyTorch .pth format)
    try:
        gen.load_state_dict(torch.load(args.model_path, map_location=device))
        gen.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: If you haven't converted the weights from MXNet yet, run convert_weights.py first.")
        return

    for path in args.images:
        print(f"Processing: {path}")
        try:
            img = dlib.load_rgb_image(path)
        except Exception as e:
            print(f"Could not load image {path}: {e}")
            continue
            
        t = time.time()
        faces = det(img, 1)
        print("face detection: %.3fs" % (time.time() - t))
        
        if len(faces) == 0:
            print(f"No face detected in {path}. Skipping...")
            continue
            
        for i, face in enumerate(faces):
            t = time.time()
            hw = max(face.right() - face.left(), face.bottom() - face.top())
            x = max(face.left() - int(0.3 * hw), 0)
            y = max(face.top() - int(0.5 * hw), 0)
            hw = int(hw * 1.6)
            
            # Crop and prepare for tensor
            raw_crop = img[y:y+hw, x:x+hw]
            if raw_crop.shape[0] == 0 or raw_crop.shape[1] == 0:
                continue
                
            pil_img = Image.fromarray(raw_crop)
            # Resize
            real = TF.resize(pil_img, args.resize, Image.BICUBIC)
            # To Tensor and Normalize
            real = TF.to_tensor(real)
            real = TF.normalize(real, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            real = real.unsqueeze(0).to(device)
            
            with torch.no_grad():
                fake, _ = gen(real)
                
            print(f"anime generation (face {i+1}): %.3fs" % (time.time() - t))
            
            # Visualization
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original (Cropped)")
            plt.imshow(raw_crop)
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.title("Anime Version")
            plt.imshow(reconstruct_color(fake[0]))
            plt.axis("off")
            plt.show()

if __name__ == "__main__":
    run_inference()
