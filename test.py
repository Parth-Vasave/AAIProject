import argparse
import torch
import matplotlib.pyplot as plt
from dataset import load_image, reconstruct_color
from pix2pix_gan import ResnetGenerator, PatchDiscriminator

def test():
    parser = argparse.ArgumentParser(description="Start a PyTorch cycle_gan tester.")
    parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
    parser.add_argument("--reversed", help="reverse transformation (B to A)", action="store_true")
    parser.add_argument("--model", help="model name prefix (default: selfie2anime)", type=str, default="selfie2anime")
    parser.add_argument("--resize", help="image resize size (default: 256)", type=int, default=256)
    parser.add_argument("--device", help="cuda, mps, or cpu", type=str, default=None)
    args = parser.parse_args()

    # Device detection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Using device: {device}")

    print("Loading models...", flush=True)
    gen_ab = ResnetGenerator().to(device)
    gen_ba = ResnetGenerator().to(device)
    
    try:
        gen_ab.load_state_dict(torch.load(f"model/{args.model}.gen_ab.pth", map_location=device))
        gen_ba.load_state_dict(torch.load(f"model/{args.model}.gen_ba.pth", map_location=device))
        gen_ab.eval()
        gen_ba.eval()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Ensure you have converted the .params weights to .pth first.")
        return

    for path in args.images:
        print(f"Testing: {path}")
        real = load_image(path, args.resize).to(device)
        
        with torch.no_grad():
            if args.reversed:
                fake, _ = gen_ba(real)
                rec, _ = gen_ab(fake)
            else:
                fake, _ = gen_ab(real)
                rec, _ = gen_ba(fake)
        
        # Display results
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(reconstruct_color(real[0]))
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.title("Fake")
        plt.imshow(reconstruct_color(fake[0]))
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.title("Reconstructed")
        plt.imshow(reconstruct_color(rec[0]))
        plt.axis("off")
        
        # Save output in Colab environments
        plt.savefig('output.png')
        plt.show()

if __name__ == "__main__":
    test()
