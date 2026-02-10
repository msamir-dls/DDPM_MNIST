import os
import torch
import torchvision
import mlflow
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_device
from src.models.unet import UNet
from src.schedulers.ddim_solver import DDIM

def load_trained_model(checkpoint_path, config):
    """Load trained DDPM model for DDIM sampling with compatibility handling"""
    device = get_device()
    
    # Initialize model (same architecture as training)
    model = UNet(config).to(device)
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load with strict=False to handle architecture differences
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"  Missing keys: {missing_keys[:5]}")  # Show first 5
    if unexpected_keys:
        print(f"  Unexpected keys: {unexpected_keys[:5]}")
    
    model.eval()
    print(f" Loaded trained model from {checkpoint_path}")
    print(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   - Loss: {checkpoint.get('loss', checkpoint.get('final_loss', 'N/A')):.4f}")
    
    return model, device

def generate_ddim_samples(model, ddim_scheduler, config, device, num_samples=16):
    """Generate samples using DDIM"""
    model.eval()
    with torch.no_grad():
        sample_shape = (num_samples, 
                       config['dataset']['channels'], 
                       config['dataset']['img_size'], 
                       config['dataset']['img_size'])
        
        print(f" Generating {num_samples} samples with DDIM...")
        samples = ddim_scheduler.sample(model, sample_shape)
        samples = (samples.clamp(-1, 1) + 1) / 2  # [0, 1]
        
        return samples

def main():
    """Main DDIM inference pipeline - Simplified version"""
    # Load DDIM config
    config = load_config("configs/ddim_inference.yaml")
    print(f"[*] DDIM Configuration:")
    print(f"   - Sampling steps: {config['ddim']['sampling_steps']}")
    print(f"   - Eta (stochasticity): {config['ddim']['eta']}")
    
    device = get_device()
    print(f"📱 Using device: {device}")
    
    # Load trained model
    checkpoint_path = config.get('model_checkpoint', 'checkpoints/ddpm_final.pth')
    model, device = load_trained_model(checkpoint_path, config)
    
    # Initialize DDIM scheduler
    ddim_scheduler = DDIM(config).to(device)
    
    # Generate samples
    num_samples = config['inference'].get('num_samples', 16)
    samples = generate_ddim_samples(model, ddim_scheduler, config, device, num_samples)
    
    # Save samples
    os.makedirs('ddim_outputs', exist_ok=True)
    
    # Save grid
    grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)
    grid_path = 'ddim_outputs/ddim_samples_grid.png'
    torchvision.utils.save_image(grid, grid_path, normalize=True)
    print(f" Saved samples grid to {grid_path}")
    
    # Save individual samples
    for i in range(min(10, len(samples))):
        sample_path = f'ddim_outputs/sample_{i:03d}.png'
        torchvision.utils.save_image(samples[i], sample_path, normalize=True)
    
    # Display samples
    try:
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray')
        plt.title(f'DDIM Generated Samples ({config["ddim"]["sampling_steps"]} steps)')
        plt.axis('off')
        plt.show()
    except:
        print("  Could not display images. Check 'ddim_outputs/' folder.")
    
    # Compare with DDPM if checkpoint exists
    try:
        from src.schedulers.gaussian import GaussianDiffusion
        import time
        
        print("\n Comparing with DDPM (10 steps for speed):")
        
        # Create DDPM scheduler
        ddpm_config = config.copy()
        ddpm_config['diffusion']['timesteps'] = 10  # Just 10 steps for quick comparison
        
        ddpm_scheduler = GaussianDiffusion(ddpm_config).to(device)
        
        # Time DDPM
        start = time.time()
        with torch.no_grad():
            ddpm_samples = ddpm_scheduler.sample(model, (4, 1, 32, 32))
        ddpm_time = time.time() - start
        
        # Time DDIM
        start = time.time()
        with torch.no_grad():
            ddim_samples = ddim_scheduler.sample(model, (4, 1, 32, 32))
        ddim_time = time.time() - start
        
        print(f"   DDPM (10 steps): {ddpm_time:.2f}s")
        print(f"   DDIM ({ddpm_scheduler.ddim_timesteps} steps): {ddim_time:.2f}s")
        print(f"   Speedup: {ddpm_time/ddim_time:.1f}x faster!")
        
    except Exception as e:
        print(f"  Could not compare with DDPM: {e}")
    
    print(f"\n DDIM Inference Complete!")
    print(f"   Generated {len(samples)} samples")
    print(f"   Check 'ddim_outputs/' folder for results")

if __name__ == "__main__":
    main()