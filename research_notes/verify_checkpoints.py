import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hashlib
import torch
import yaml
from model_jta import create_model as create_model_jta
from model_jrdb import create_model as create_model_jrdb
from utils.utils import create_logger

def get_file_hash(path):
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()

def verify_checkpoint(path):
    print("=" * 80)
    print(f"VERIFYING CHECKPOINT: {path}")
    print("=" * 80)
    abs_path = os.path.abspath(path)
    file_size = os.path.getsize(abs_path)
    sha256 = get_file_hash(abs_path)

    print(f"1. Absolute Path: {abs_path}")
    print(f"2. File Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    print(f"   SHA-256 Hash: {sha256}")

    # 3. Load on CPU
    ckpt = torch.load(abs_path, map_location=torch.device('cpu'))
    print("3. Checkpoint loaded successfully on CPU.")

    # 4. Dictionary keys
    keys = list(ckpt.keys())
    print(f"4. Checkpoint Dictionary Keys: {keys}")

    # 5. Embedded Configuration
    config = ckpt.get('config', {})
    print(f"5. Embedded Configuration:")
    print(yaml.dump(config, default_flow_style=False, indent=2))

    # 6. Identify Dataset Identity
    train_datasets = config.get('DATA', {}).get('train_datasets', [])
    dataset_name = train_datasets[0] if train_datasets else None
    print(f"6. Detected Dataset Identity from Config: {dataset_name}")

    # 7. Model Compatibility Verification
    logger = create_logger('')
    config["DEVICE"] = "cpu"
    
    if dataset_name == 'jta_all_visual_cues':
        model = create_model_jta(config, logger)
        model_type = "JTA Model (model_jta.py)"
    elif dataset_name == 'jrdb_2dbox':
        model = create_model_jrdb(config, logger)
        model_type = "JRDB Model (model_jrdb.py)"
    else:
        raise ValueError(f"Unknown dataset identity: {dataset_name}")

    print(f"7. Initialized Corresponding Architecture: {model_type}")

    # 8. Inspect Model State Dict
    model_state = ckpt.get('model', {})
    print(f"8. Model State Dict contains {len(model_state)} parameter tensors.")

    # 9. Missing or Unexpected Keys
    expected_keys = set(model.state_dict().keys())
    ckpt_keys = set(model_state.keys())

    missing_keys = expected_keys - ckpt_keys
    unexpected_keys = ckpt_keys - expected_keys

    print(f"9. Parameter Key Validation:")
    print(f"   Missing Keys: {len(missing_keys)} -> {missing_keys}")
    print(f"   Unexpected Keys: {len(unexpected_keys)} -> {unexpected_keys}")

    load_result = model.load_state_dict(model_state, strict=True)
    print(f"   Strict load_state_dict result: SUCCESS (all keys matched identically)")

    # 10. Number of Model Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"10. Model Parameters:")
    print(f"    Total Parameters: {total_params:,}")
    print(f"    Trainable Parameters: {trainable_params:,}\n")

if __name__ == '__main__':
    checkpoints = [
        "checkpoints_data/jta/checkpoint/checkpoint.pth.tar",
        "checkpoints_data/jrdb/checkpoint/checkpoint.pth.tar"
    ]
    for cp in checkpoints:
        verify_checkpoint(cp)
