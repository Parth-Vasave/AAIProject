import os
import torch
import numpy as np

# --- PATCH FOR GOOGLE COLAB / NUMPY 2.x COMPATIBILITY ---
# MXNet is incompatible with NumPy 2.0. We patch it here before importing.
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'bool_'):
    np.bool_ = bool
# --------------------------------------------------------

try:
    import mxnet as mx
except ImportError:
    print("Error: mxnet is not installed. Run: !pip install mxnet")
    exit(1)

from pix2pix_gan import ResnetGenerator

def convert_mxnet_to_pytorch(mxnet_params_path, output_pth_path):
    print(f"Loading MXNet parameters from {mxnet_params_path}...")
    try:
        # Load MXNet ndarray dictionary
        mx_params = mx.nd.load(mxnet_params_path)
    except Exception as e:
        print(f"Error: Could not load MXNet parameters. Make sure 'mxnet' is installed.")
        print(f"Detail: {e}")
        return

    # Create the PyTorch model
    model = ResnetGenerator()
    state_dict = model.state_dict()
    
    print("Mapping weights...")
    
    # This mapping is more robust. We look for all weights and internal SN vectors.
    # MXNet stores SN vectors as '_u', PyTorch as 'weight_u'.
    mx_keys = [k for k in mx_params.keys()]
    pt_keys = [k for k in state_dict.keys()]
    
    print(f"Total MXNet parameters: {len(mx_keys)}")
    print(f"Total PyTorch parameters: {len(pt_keys)}")
    
    # We map by identifying the type of parameter and its position in the sequence.
    mx_weights = sorted([k for k in mx_keys if 'weight' in k])
    mx_u = sorted([k for k in mx_keys if '_u' in k])
    
    # PyTorch weights are either inside 'spectral_norm' (weight_orig) or regular (weight)
    pt_weights = sorted([k for k in pt_keys if k.endswith('.weight_orig') or (k.endswith('.weight') and 'weight_u' not in k)])
    pt_u = sorted([k for k in pt_keys if k.endswith('.weight_u')])

    if len(mx_weights) != len(pt_weights):
        print(f"⚠️ Warning: Weight count mismatch! MX={len(mx_weights)}, PT={len(pt_weights)}")
    
    # Map weights sequentially
    for i in range(min(len(mx_weights), len(pt_weights))):
        mx_val = mx_params[mx_weights[i]].asnumpy()
        state_dict[pt_weights[i]] = torch.from_numpy(mx_val)
        
    # Map spectral norm 'u' vectors sequentially
    for i in range(min(len(mx_u), len(pt_u))):
        mx_val = mx_params[mx_u[i]].asnumpy()
        state_dict[pt_u[i]] = torch.from_numpy(mx_val)

    # Save the new state dict
    torch.save(state_dict, output_pth_path)
    print(f"Successfully saved PyTorch weights to {output_pth_path}")

if __name__ == "__main__":
    # Automatically Detect all .params files in the model folder
    model_dir = "model"
    if not os.path.exists(model_dir):
        print(f"Error: {model_dir} directory not found.")
        exit(1)
        
    param_files = [f for f in os.listdir(model_dir) if f.endswith(".params")]
    
    if not param_files:
        print(f"No .params files found in {model_dir}/")
        exit(1)
        
    for p_file in param_files:
        mx_path = os.path.join(model_dir, p_file)
        pt_path = mx_path.replace(".params", ".pth")
        print(f"\n--- Converting {p_file} ---")
        convert_mxnet_to_pytorch(mx_path, pt_path)
    
    print("\n✅ All conversions complete.")
