import scipy.io as scio
import sys
import numpy as np

def inspect(path):
    print(f"Inspecting: {path}")
    data = scio.loadmat(path)
    print("Keys:", data.keys())
    for k, v in data.items():
        if k.startswith('__'): continue
        if isinstance(v, np.ndarray):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{k}: {type(v)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        print("Please provide a file path.")
