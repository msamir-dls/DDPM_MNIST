# ddim_inference.py
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
    """Load trained DDPM model for DDIM sampling"""
    device = get_device()
    
    # Initialize model (same architecture as training)
    model = UNet(config).to(device)
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f" Loaded trained model from {checkpoint_path}")
    print(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   - Loss: {checkpoint.get('loss', checkpoint.get('final_loss', 'N/A')):.4f}")
    
    return model, device

def compare_ddpm_ddim(model, ddpm_scheduler, ddim_scheduler, config, device, num_samples=8):
    """Compare DDPM vs DDIM sampling quality and speed"""
    import time
    
    results = {}
    
    for method_name, scheduler in [("DDPM", ddpm_scheduler), ("DDIM", ddim_scheduler)]:
        print(f"\n  Testing {method_name} sampling...")
        
        # Time the sampling
        start_time = time.time()
        
        with torch.no_grad():
            sample_shape = (num_samples, 
                          config['dataset']['channels'], 
                          config['dataset']['img_size'], 
                          config['dataset']['img_size'])
            
            samples, intermediates = scheduler.sample(
                model, 
                sample_shape, 
                return_intermediates=True
            )
            samples = (samples.clamp(-1, 1) + 1) / 2  # [0, 1]
        
        sampling_time = time.time() - start_time
        
        # Save results
        results[method_name] = {
            'samples': samples,
            'time': sampling_time,
            'steps': scheduler.timesteps if method_name == 'DDPM' else scheduler.ddim_timesteps,
            'intermediates': intermediates
        }
        
        print(f"    Generated {num_samples} samples in {sampling_time:.2f}s")
        print(f"    Steps: {results[method_name]['steps']}")
    
    return results

def visualize_comparison(results, config):
    """Visualize DDPM vs DDIM comparison"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Plot samples
    for idx, (method_name, result) in enumerate(results.items()):
        samples = result['samples'][:4]  # First 4 samples
        
        for i in range(4):
            ax = axes[idx, i]
            ax.imshow(samples[i].cpu().permute(1, 2, 0).squeeze(), cmap='gray')
            ax.set_title(f'{method_name} - Sample {i+1}')
            ax.axis('off')
    
    plt.suptitle(f"DDPM vs DDIM Comparison\n"
                f"DDPM: {results['DDPM']['steps']} steps, {results['DDPM']['time']:.2f}s\n"
                f"DDIM: {results['DDIM']['steps']} steps, {results['DDIM']['time']:.2f}s\n"
                f"Speedup: {results['DDPM']['time']/results['DDIM']['time']:.1f}x faster", 
                fontsize=14)
    plt.tight_layout()
    plt.savefig('ddpm_vs_ddim.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return 'ddpm_vs_ddim.png'

def generate_ddim_samples_grid(model, ddim_scheduler, config, device, 
                               steps_list=[1000, 500, 250, 100, 50, 25, 10]):
    """Generate samples with different DDIM step counts"""
    samples_grid = []
    
    for steps in steps_list:
        # Update DDIM steps
        ddim_scheduler.ddim_timesteps = steps
        ddim_scheduler.ddim_timestep_sequence = ddim_scheduler.make_ddim_timesteps()
        
        print(f" Generating with {steps} DDIM steps...")
        
        with torch.no_grad():
            sample_shape = (4, config['dataset']['channels'], 
                          config['dataset']['img_size'], 
                          config['dataset']['img_size'])
            
            samples = ddim_scheduler.sample(model, sample_shape)
            samples = (samples.clamp(-1, 1) + 1) / 2
            samples_grid.append(samples)
    
    # Create comparison grid
    grid_rows = []
    for i, (steps, samples) in enumerate(zip(steps_list, samples_grid)):
        row = torchvision.utils.make_grid(samples, nrow=4, padding=2)
        # Add step count label
        row_with_label = torch.zeros((3, row.shape[1] + 30, row.shape[2]))
        row_with_label[:, 30:, :] = row
        # Simpler: just create a title
        grid_rows.append(row)
    
    final_grid = torch.cat(grid_rows, dim=1)  # Stack vertically
    
    # Save
    torchvision.utils.save_image(final_grid, 'ddim_steps_comparison.png', normalize=True)
    
    # Plot with matplotlib for better labels
    fig, axes = plt.subplots(len(steps_list), 1, figsize=(12, 3*len(steps_list)))
    
    for idx, (steps, samples) in enumerate(zip(steps_list, samples_grid)):
        grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)
        axes[idx].imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
        axes[idx].set_title(f'DDIM with {steps} steps', fontsize=12)
        axes[idx].axis('off')
    
    plt.suptitle('DDIM Sampling Quality vs Number of Steps', fontsize=16)
    plt.tight_layout()
    plt.savefig('ddim_steps_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return 'ddim_steps_comparison.png'

def main():
    """Main DDIM inference pipeline"""
    # Load DDIM config
    config = load_config("configs/ddim_inference.yaml")
    print(f"[*] DDIM Configuration:")
    print(f"   - Sampling steps: {config['ddim']['sampling_steps']}")
    print(f"   - Eta (stochasticity): {config['ddim']['eta']}")
    
    device = get_device()
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name=config['run_name']):
        # Log DDIM parameters
        mlflow.log_params({
            'sampling_steps': config['ddim']['sampling_steps'],
            'eta': config['ddim']['eta'],
            'num_samples': config['inference']['num_samples'],
            'batch_size': config['inference']['batch_size']
        })
        
        # Load trained model
        model, device = load_trained_model(config['model_checkpoint'], config)
        
        # Initialize schedulers
        from src.schedulers.gaussian import GaussianDiffusion
        ddpm_scheduler = GaussianDiffusion(config).to(device)
        ddim_scheduler = DDIM(config).to(device)
        
        print(f"\n Comparing DDPM vs DDIM...")
        
        # 1. Direct comparison
        results = compare_ddpm_ddim(
            model, ddpm_scheduler, ddim_scheduler, 
            config, device, num_samples=8
        )
        
        # Visualize comparison
        comparison_img = visualize_comparison(results, config)
        mlflow.log_artifact(comparison_img)
        
        # Log timing metrics
        mlflow.log_metric("ddpm_sampling_time", results['DDPM']['time'])
        mlflow.log_metric("ddim_sampling_time", results['DDIM']['time'])
        mlflow.log_metric("speedup_factor", results['DDPM']['time'] / results['DDIM']['time'])
        
        # 2. Generate batch of DDIM samples
        print(f"\n Generating {config['inference']['num_samples']} DDIM samples...")
        
        all_samples = []
        num_batches = (config['inference']['num_samples'] + 
                      config['inference']['batch_size'] - 1) // config['inference']['batch_size']
        
        for batch_idx in range(num_batches):
            current_batch = min(config['inference']['batch_size'], 
                              config['inference']['num_samples'] - batch_idx * config['inference']['batch_size'])
            
            with torch.no_grad():
                sample_shape = (current_batch, 
                              config['dataset']['channels'], 
                              config['dataset']['img_size'], 
                              config['dataset']['img_size'])
                
                samples = ddim_scheduler.sample(model, sample_shape)
                samples = (samples.clamp(-1, 1) + 1) / 2
                all_samples.append(samples.cpu())
            
            print(f"   Batch {batch_idx+1}/{num_batches}: {current_batch} samples")
        
        all_samples = torch.cat(all_samples, dim=0)
        
        # Save samples grid
        grid = torchvision.utils.make_grid(all_samples[:64], nrow=8, padding=2)
        torchvision.utils.save_image(grid, 'ddim_samples_grid.png', normalize=True)
        mlflow.log_artifact('ddim_samples_grid.png')
        
        # 3. Step count comparison
        print(f"\n Testing DDIM with different step counts...")
        steps_comparison_img = generate_ddim_samples_grid(
            model, ddim_scheduler, config, device,
            steps_list=[1000, 500, 250, 100, 50, 25, 10]
        )
        mlflow.log_artifact(steps_comparison_img)
        
        # 4. Save some samples individually
        os.makedirs('ddim_outputs', exist_ok=True)
        for i in range(min(16, len(all_samples))):
            torchvision.utils.save_image(
                all_samples[i], 
                f'ddim_outputs/sample_{i:03d}.png',
                normalize=True
            )
        
        mlflow.log_artifact('ddim_outputs/')
        
        print(f"\n DDIM Inference Complete!")
        print(f"   - Generated {len(all_samples)} samples")
        print(f"   - Speedup: {results['DDPM']['time']/results['DDIM']['time']:.1f}x faster than DDPM")
        print(f"   - Check MLflow for results: {config['mlflow']['tracking_uri']}")

if __name__ == "__main__":
    main()