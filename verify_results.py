import scipy.io as scio
import numpy as np

data = scio.loadmat('experiment_results.mat')
print("Keys:", data.keys())
for key in ['ssim_setting_A', 'ssim_setting_B1', 'ssim_setting_B2', 'fid_setting_A', 'fid_setting_B1', 'fid_setting_B2']:
    if key in data:
        print(f"{key} shape: {data[key].shape}")
        print(f"{key} sample: {data[key][0, :5]}")
    else:
        print(f"{key} missing!")
