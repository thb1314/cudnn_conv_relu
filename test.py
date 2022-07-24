import numpy as np
import torch.nn.functional as F
import torch

def _load_tensor(file):     
    with open(file, "rb") as f:
        binary_data = f.read()

    magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
    assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."
    
    dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)

    if dtype == 0:
        np_dtype = np.float32
    elif dtype == 1:
        np_dtype = np.float16
    else:
        assert False, f"Unsupport dtype = {dtype}, can not convert to numpy dtype"
        
    return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)


def load_tensor(file):
    if file.endswith("npz"):
        return np.load(file)['data']
    elif file.endswith("npy"):
        return np.load(file)
    else:
        return _load_tensor(file)


def test():
    input_tensor = load_tensor('q_tensor.npz')
    weight_tensor = load_tensor('kernel_tensor.npz')
    bias_tensor = load_tensor('bias_tensor.npz')
    out_tensor = load_tensor('out_tensor.npz')
    
    input_tensor = torch.as_tensor(input_tensor).float()
    weight_tensor = torch.as_tensor(weight_tensor).float()
    bias_tensor = torch.as_tensor(bias_tensor).float()
    out_tensor = torch.as_tensor(out_tensor).float()
    
    print(input_tensor.shape)
    print(weight_tensor.shape)
    print(bias_tensor.shape)
    print(out_tensor.shape)
    # 关键方法 F.conv2d

    out_tensor_torch = F.conv2d(input_tensor, weight_tensor, bias_tensor, stride=1)
    out_tensor_torch = F.relu(out_tensor_torch)
    # out_tensor_torch = torch.nn.SiLU(out_tensor_torch)

    print(torch.abs(out_tensor_torch - out_tensor).max())


if __name__ == "__main__":
    test()
