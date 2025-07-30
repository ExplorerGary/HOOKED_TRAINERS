# helper_modules.py
import os
import torch
import numpy as np
import logging

def scan_pt(base_dir:str):
    '''
    扫描特定盘下所有的.pt文件，然后返回一个整理好的绝对路径列表
    '''
    avail_pt = []

    for file in os.listdir(base_dir):
        # if file.endswith("pt"):
        if file.endswith(".pt"):
            file_path = os.path.join(base_dir,file)
            avail_pt.append(file_path)

    return avail_pt

def read_pt_tensor(pt_path) -> torch.Tensor:
    '''
    返回 array ，等待使用 EG 编码。
        如果 pt 是 dict，则集体展平成一个一维 torch.Tensor 叫做pt_array。然后返回。
    '''
    pt = torch.load(pt_path, map_location="cpu")
    if isinstance(pt, dict):
        tensors = []
        for v in pt.values():
            tensors.append(v.flatten())
        # print(f"array:{len(arrays)}")
        pt_array = torch.cat(tensors)
    else:
        pt_array = pt.flatten()
    return pt_array

def entropy(pt_array):
    values, counts = np.unique(pt_array, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def quantlization_fuct(flat_tensor:torch.Tensor,
                       scaling:int = int(1e6),
                       fp64_enable:bool = False):
    '''
    观察记录：
    1. fp16的最高数字约为为6.5e5，也就意味着我们最好不要使用1e3及以上的tensor，不然就变成inf了(因为有情况下时会出现Xe2的数量级的)
        但不知道为什么，经过测试后发现原来的scaling是可行的。
    
    '''
    
    
    try:
        if fp64_enable:
            flat_tensor = flat_tensor.to(dtype=torch.float64)
            
        quantilized = torch.round(flat_tensor * scaling) / scaling
        
        return quantilized
    
    
    except Exception as e:
        raise e

import torch

def eval(a,b):
    a_nan = torch.isnan(a).any()
    b_nan = torch.isnan(b).any()
    a_inf = torch.isinf(a).any()
    b_inf = torch.isinf(b).any()

    return a_nan,b_nan,a_inf,b_inf

x = torch.ones(3, dtype=torch.float16)
base_dir = "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files"
avail_pt = scan_pt(base_dir=base_dir)
scaling = 1e6 # 2**8
for pt in avail_pt:
    x = read_pt_tensor(pt)
    a = x * scaling
    b = x * int(scaling)
    a_nan,b_nan,a_inf,b_inf = eval(a,b)
    a_e = entropy(a)
    b_e = entropy(b)
    
    print(a_nan,b_nan,a_inf,b_inf)
    print(a_e,b_e)
    print("x * 1e6:", (x * scaling).dtype)          
    print("x * int(1e6):", (x * int(scaling)).dtype)  

    print("are they the same?", torch.equal(a,b))
    print(a,b)
    print()



