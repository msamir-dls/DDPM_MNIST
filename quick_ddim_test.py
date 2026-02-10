# quick_ddim_test.py
import torch
import sys
import os
sys.path.append('.')

from src.models.unet import UNet
from src.schedulers.ddim_solver import DDIM
from src.utils import load_config

def quick_test():
    """Quick test to verify DDIM works with your trained model"""
    print(" Quick DDIM Test")
    print("=" * 50)
    
    # Load config
    config = load_config("configs/ddpm_train.yaml")
    
    # Add DDIM settings
    config['ddim'] = {
        'sampling_steps': 50,
        'eta': 0.0
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = UNet(config).to(device)
    checkpoint = torch.load('checkpoints/ddpm_final.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize DDIM
    ddim = DDIM(config).to(device)
    
    # Generate samples
    print(f" Testing DDIM with {ddim.ddim_timesteps} steps...")
    
    with torch.no_grad():
        sample_shape = (4, 1, 32, 32)
        samples = ddim.sample(model, sample_shape)
        samples = (samples.clamp(-1, 1) + 1) / 2
        
        # Save
        import torchvision
        torchvision.utils.save_image(samples, 'ddim_quick_test.png', nrow=2)
    
    print(f" DDIM test complete! Check 'ddim_quick_test.png'")
    
    # Compare with DDPM
    from src.schedulers.gaussian import GaussianDiffusion
    ddpm = GaussianDiffusion(config).to(device)
    
    print(f"\n Speed comparison:")
    import time
    
    # DDIM timing
    start = time.time()
    with torch.no_grad():
        _ = ddim.sample(model, sample_shape)
    ddim_time = time.time() - start
    
    # DDPM timing
    start = time.time()
    with torch.no_grad():
        _ = ddpm.sample(model, sample_shape)
    ddpm_time = time.time() - start
    
    print(f"   DDPM ({ddpm.timesteps} steps): {ddpm_time:.2f}s")
    print(f"   DDIM ({ddim.ddim_timesteps} steps): {ddim_time:.2f}s")
    print(f"   Speedup: {ddpm_time/ddim_time:.1f}x faster!")
    
    return samples

if __name__ == "__main__":
    quick_test()