
import os
import torch
import numpy as np
import scipy.io as scio
from tqdm import tqdm
import shutil
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import project modules
from tfdiff.params import all_params, AttrDict
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.diffusion import GaussianDiffusion, SignalDiffusion
from tfdiff.dataset import from_path_inference, _nested_map
from experiment import eval_ssim, calculate_fid, save_wifi

import os
import torch
import numpy as np
import scipy.io as scio
from tqdm import tqdm
import shutil
import warnings
import argparse

# Suppress warnings
warnings.filterwarnings("ignore")

# Import project modules
from tfdiff.params import all_params, AttrDict
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.diffusion import GaussianDiffusion, SignalDiffusion
from tfdiff.dataset import from_path_inference, _nested_map
from experiment import eval_ssim, calculate_fid, save_wifi

def run_reproduction(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting RF-Diffusion Reproduction (Inference & Data Saving) ===")
    print(f"Device: {device}")
    print(f"Number of FID runs scheduled: {args.n_fid_runs}")
    
    # 1. Setup Parameters
    task_id = 0  # WiFi
    params = all_params[task_id]
    
    # Paths
    params.model_dir = './model/wifi/b32-256-100s'
    # Important: cond_dir must be a list of paths, otherwise it iterates over the characters of the string
    params.cond_dir = ['./dataset/wifi']
    params.inference_batch_size = 1
    
    # Output Directories
    img_save_root = './dataset/wifi/img/reproduction'
    dir_gt = os.path.join(img_save_root, 'gt')
    dir_pred = os.path.join(img_save_root, 'pred')
    
    # Directory for saving data for plotting
    plot_data_dir = './plots/data'
    plot_file_ssim = os.path.join(plot_data_dir, 'reproduction_ssim_wifi.mat')
    plot_file_fid = os.path.join(plot_data_dir, 'reproduction_fid_wifi.mat')
    
    os.makedirs(plot_data_dir, exist_ok=True)
    
    # 2. Load Model
    print(f"Loading model from {params.model_dir}...")
    weights_path = os.path.join(params.model_dir, 'weights.pt')
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        return

    checkpoint = torch.load(weights_path, map_location=device)
    model = tfdiff_WiFi(AttrDict(params)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.params.override(params)
    
    diffusion = SignalDiffusion(params) if params.signal_diffusion else GaussianDiffusion(params)
    
    # 3. Load Dataset
    print(f"Loading dataset from {params.cond_dir}...")
    # NOTE: The dataset loader might be slow if there are many files.
    dataset = from_path_inference(params)
    print(f"Dataset loaded. Total samples per run: {len(dataset)}")
    
    # Storage for results
    all_fid_scores = []
    # We only need one full set of SSIM scores (frequency distribution of samples)
    # But we can capture it from the first run.
    first_run_ssim_scores = []
    
    # 4. Multi-Run Loop
    for run_idx in range(args.n_fid_runs):
        print(f"\n--- Starting Run {run_idx + 1}/{args.n_fid_runs} ---")
        
        # Clean previous prediction images to obtain a fresh FID calculation for this run
        if os.path.exists(dir_pred):
            shutil.rmtree(dir_pred)
        # We can keep GT images if they are already generated, but to be safe and simple, clear all
        # or just clear pred. GT is deterministic, so we only need to generate it once technically,
        # but save_wifi generates both. Let's just clear both for simplicity to avoid mismatched files.
        if os.path.exists(img_save_root):
            shutil.rmtree(img_save_root)
        os.makedirs(dir_gt, exist_ok=True)
        os.makedirs(dir_pred, exist_ok=True)
        
        current_run_ssims = []
        cur_index = 0
        
        with torch.no_grad():
            for features in tqdm(dataset, desc=f"Run {run_idx+1} Inference"):
                features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                data = features["data"] # Ground Truth
                cond = features["cond"] # Condition
                
                # Sampling
                pred = diffusion.native_sampling(model, data, cond, device)
                
                # Split batch
                data_samples = [torch.view_as_complex(s) for s in torch.split(data, 1, dim=0)]
                pred_samples = [torch.view_as_complex(s) for s in torch.split(pred, 1, dim=0)]
                cond_samples = [torch.view_as_complex(s) for s in torch.split(cond, 1, dim=0)]
                
                for b, p_sample in enumerate(pred_samples):
                    d_sample = data_samples[b]
                    
                    # Calculate SSIM (Per Sample)
                    # We store this for every run, but will only save the first run's distribution for the plot
                    cur_ssim = eval_ssim(p_sample, d_sample, params.sample_rate, params.input_dim, device=device)
                    current_run_ssims.append(cur_ssim.item())
                    
                    # Save images (GT and Pred) for FID
                    # We pass a dummy output dir for mat files to avoid clutter
                    save_wifi(
                        "./dataset/wifi/output/temp_repro", 
                        d_sample.cpu().detach(),
                        p_sample.cpu().detach(),
                        cond_samples[b].cpu().detach(),
                        cur_index,
                        0,
                        img_data_dir=dir_gt,
                        img_pred_dir=dir_pred
                    )
                    cur_index += 1
        
        # Store SSIMs if it's the first run
        if run_idx == 0:
            first_run_ssim_scores = current_run_ssims
            print(f"Run 1 SSIM Mean: {np.mean(first_run_ssim_scores):.4f}")
            
        # Calculate FID for this run
        print(f"Calculating FID for Run {run_idx + 1}...")
        # Using correction factor 1.9 for WiFi as per original code
        fid_val = calculate_fid(dir_pred, dir_gt, device, corr=1.9)
        all_fid_scores.append(fid_val)
        print(f"Run {run_idx + 1} FID: {fid_val:.4f}")

    # 5. Save Results
    print("\n" + "="*50)
    print("SAVING RESULTS")
    
    # Save SSIM (Distribution of samples from single run)
    # Using 'data_wifi_repro_ssim' key to differentiate, or match 'data_wifi_sigma' format if you prefer
    scio.savemat(plot_file_ssim, {'repro_ssim': np.array(first_run_ssim_scores).reshape(-1, 1)})
    print(f"Saved SSIM data to: {plot_file_ssim}")
    
    # Save FID (Distribution of experiment runs)
    scio.savemat(plot_file_fid, {'repro_fid': np.array(all_fid_scores).reshape(-1, 1)})
    print(f"Saved FID data to:  {plot_file_fid}")
    
    print("\nSummary:")
    print(f"SSIM (Mean of {len(first_run_ssim_scores)} samples): {np.mean(first_run_ssim_scores):.4f}")
    print(f"FID  (Mean of {len(all_fid_scores)} runs)   : {np.mean(all_fid_scores):.4f}")
    print("="*50)
    
    # Cleanup temp dirs
    if os.path.exists("./dataset/wifi/output/temp_repro"):
        shutil.rmtree("./dataset/wifi/output/temp_repro")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce RF-Diffusion WiFi Results")
    parser.add_argument("--n_fid_runs", type=int, default=5, help="Number of times to repeat the experiment for FID distribution")
    args = parser.parse_args()
    
    run_reproduction(args)
