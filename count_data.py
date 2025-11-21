import os
from glob import glob
from tfdiff.params import all_params
from tfdiff.dataset import WiFiDataset, FMCWDataset, MIMODataset, EEGDataset

def count_data():
    task_names = ["WiFi", "FMCW", "MIMO", "EEG"]
    
    for task_id, params in enumerate(all_params):
        print(f"--- Task: {task_names[task_id]} (ID: {task_id}) ---")
        
        # Training Data
        data_dir = params.data_dir
        if task_id == 0:
            train_dataset = WiFiDataset(data_dir)
        elif task_id == 1:
            train_dataset = FMCWDataset(data_dir)
        elif task_id == 2:
            train_dataset = MIMODataset(data_dir)
        elif task_id == 3:
            train_dataset = EEGDataset(data_dir)
            
        print(f"Training Data Count: {len(train_dataset)}")
        
        # Inference Data
        cond_dir = params.cond_dir
        if task_id == 0:
            infer_dataset = WiFiDataset(cond_dir)
        elif task_id == 1:
            infer_dataset = FMCWDataset(cond_dir)
        elif task_id == 2:
            infer_dataset = MIMODataset(cond_dir)
        elif task_id == 3:
            infer_dataset = EEGDataset(cond_dir)
            
        print(f"Inference Data Count: {len(infer_dataset)}")
        print("")

if __name__ == "__main__":
    count_data()
