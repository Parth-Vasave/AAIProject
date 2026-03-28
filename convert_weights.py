import torch
import mxnet as mx
import os
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
    
    # This mapping is approximate and based on the sequential nature of both models.
    # MXNet Gluon names parameters by the order they were created.
    # PyTorch names them by the module names we assigned.
    
    # Filter for weights and u vectors (spectral norm)
    mx_keys = sorted([k for k in mx_params.keys() if 'weight' in k or '_u' in k])
    pt_keys = sorted([k for k in state_dict.keys() if 'weight' in k or '_u' in k])
    
    # Note: PyTorch spectral_norm uses 'weight_u' and 'weight_v'. 
    # MXNet implementation used 'u'.
    
    # We will try a heuristic based on the order of layers.
    # This is often successful for Sequential models.
    
    mx_to_pt = {}
    mx_idx = 0
    
    # Separate types
    mx_weights = [k for k in mx_keys if 'weight' in k]
    mx_u = [k for k in mx_keys if '_u' in k]
    
    pt_weights = [k for k in pt_keys if 'weight' in k and 'orig' in k] # PyTorch SN stores original weight in 'weight_orig'
    pt_u = [k for k in pt_keys if 'weight_u' in k]
    
    if len(mx_weights) != len(pt_weights):
        print(f"Warning: Count mismatch! MXNet weights: {len(mx_weights)}, PyTorch weights: {len(pt_weights)}")
    
    # Map weights
    for i in range(min(len(mx_weights), len(pt_weights))):
        mx_val = mx_params[mx_weights[i]].asnumpy()
        state_dict[pt_weights[i]] = torch.from_numpy(mx_val)
        
    # Map spectral norm 'u' vectors
    for i in range(min(len(mx_u), len(pt_u))):
        mx_val = mx_params[mx_u[i]].asnumpy()
        state_dict[pt_u[i]] = torch.from_numpy(mx_val)

    # Save the new state dict
    torch.save(state_dict, output_pth_path)
    print(f"Successfully saved PyTorch weights to {output_pth_path}")

if __name__ == "__main__":
    mx_path = "model/selfie2anime.gen_ab.params"
    pt_path = "model/selfie2anime.gen_ab.pth"
    
    if os.path.exists(mx_path):
        convert_mxnet_to_pytorch(mx_path, pt_path)
    else:
        print(f"Source file {mx_path} not found.")
