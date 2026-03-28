import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CycleGANDataset, reconstruct_color
from pix2pix_gan import ResnetGenerator, PatchDiscriminator
from image_pool import ImagePool

def train():
    parser = argparse.ArgumentParser(description="Start a PyTorch CycleGAN trainer.")
    parser.add_argument("--model", help="model prefix (default: selfie2anime)", type=str, default="selfie2anime")
    parser.add_argument("--data_root", help="path to 'data' directory", type=str, default="data")
    parser.add_argument("--start_epoch", help="start epoch (default: 0)", type=int, default=0)
    parser.add_argument("--max_epochs", help="max epochs (default: 200)", type=int, default=200)
    parser.add_argument("--lr", help="learning rate (default: 0.0002)", type=float, default=0.0002)
    parser.add_argument("--batch_size", help="batch size (default: 1)", type=int, default=1)
    parser.add_argument("--lambda_cyc", help="lambda of cycle loss (default: 10.0)", type=float, default=10.0)
    parser.add_argument("--lambda_idt", help="lambda of identity loss (default: 0.5)", type=float, default=0.5)
    parser.add_argument("--device", help="cuda, mps, or cpu", type=str, default=None)
    args = parser.parse_args()

    # Device detection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Data loading
    print("Loading dataset...", flush=True)
    dataset_a = CycleGANDataset(args.data_root, args.model, "trainA")
    dataset_b = CycleGANDataset(args.data_root, args.model, "trainB")
    
    # Simple strategy: use the length of the smaller dataset
    min_len = min(len(dataset_a), len(dataset_b))
    loader_a = DataLoader(dataset_a, batch_size=args.batch_size, shuffle=True, num_workers=2)
    loader_b = DataLoader(dataset_b, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Models
    gen_ab = ResnetGenerator().to(device)
    gen_ba = ResnetGenerator().to(device)
    dis_a = PatchDiscriminator().to(device)
    dis_b = PatchDiscriminator().to(device)

    # Optimizers
    optimizer_G = optim.Adam(
        list(gen_ab.parameters()) + list(gen_ba.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(dis_a.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(dis_b.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Losses
    criterion_GAN = nn.MSELoss() # Standard for LSGAN
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Pools
    fake_a_pool = ImagePool(50)
    fake_b_pool = ImagePool(50)

    print("Starting training loop...", flush=True)
    for epoch in range(args.start_epoch, args.max_epochs):
        start_time = time.time()
        
        # Zip loaders to iterate over both datasets
        # Note: They may have different sizes, zip will stop at the shorter one.
        for i, (real_a, real_b) in enumerate(zip(loader_a, loader_b)):
            real_a, real_b = real_a.to(device), real_b.to(device)
            
            # --- Train Generators ---
            optimizer_G.zero_grad()
            
            # Identity loss
            loss_id_a = criterion_identity(gen_ba(real_a)[0], real_a) * args.lambda_cyc * args.lambda_idt
            loss_id_b = criterion_identity(gen_ab(real_b)[0], real_b) * args.lambda_cyc * args.lambda_idt
            
            # GAN loss
            fake_b, cam_b = gen_ab(real_a)
            pred_fake_b, _ = dis_b(fake_b)
            loss_GAN_ab = criterion_GAN(pred_fake_b, torch.ones_like(pred_fake_b))
            
            fake_a, cam_a = gen_ba(real_b)
            pred_fake_a, _ = dis_a(fake_a)
            loss_GAN_ba = criterion_GAN(pred_fake_a, torch.ones_like(pred_fake_a))
            
            # Cycle loss
            rec_a, _ = gen_ba(fake_b)
            loss_cycle_a = criterion_cycle(rec_a, real_a) * args.lambda_cyc
            
            rec_b, _ = gen_ab(fake_a)
            loss_cycle_b = criterion_cycle(rec_b, real_b) * args.lambda_cyc
            
            # Total G loss
            loss_G = loss_GAN_ab + loss_GAN_ba + loss_cycle_a + loss_cycle_b + loss_id_a + loss_id_b
            loss_G.backward()
            optimizer_G.step()
            
            # --- Train Discriminator A ---
            optimizer_D_A.zero_grad()
            
            # Real loss
            pred_real, _ = dis_a(real_a)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # Fake loss (from pool)
            fake_a_val = fake_a_pool.query(fake_a.detach())
            pred_fake, _ = dis_a(fake_a_val)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # --- Train Discriminator B ---
            optimizer_D_B.zero_grad()
            
            # Real loss
            pred_real, _ = dis_b(real_b)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # Fake loss (from pool)
            fake_b_val = fake_b_pool.query(fake_b.detach())
            pred_fake, _ = dis_b(fake_b_val)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()
            
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{args.max_epochs}] [Batch {i}] [G loss: {loss_G.item():.4f}] [D loss: {(loss_D_A + loss_D_B).item():.4f}]")

        # Save Checkpoints
        os.makedirs("model", exist_ok=True)
        torch.save(gen_ab.state_dict(), f"model/{args.dataset}.gen_ab.pth")
        torch.save(gen_ba.state_dict(), f"model/{args.dataset}.gen_ba.pth")
        print(f"End of epoch {epoch} | Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    train()
