import copy
import os
import sys
import argparse
from functools import partial
import time
import logging
import json
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from collections import OrderedDict
from utils.glue_data_loader import GLUEDataLoader, glue_data_metrics_map
from utils.metrics import compute_metrics
from utils.customized_trainers import CustomizedTrainer
from utils.utils import set_random_seed
from model_merging_methods.merging_methods import MergingMethod
from inference_plms_glue import dataset_model_learning_rate_mapping_dict
from utils.load_config import cache_dir
from model_merging_methods.task_vector import *
import math
import torch.nn.functional as F

parser = argparse.ArgumentParser("Interface for merging roberta models on glue")
parser.add_argument("--language_model_name", type=str, default="roberta-base", help="name of the language model", choices=["roberta-base"])
parser.add_argument("--merging_method_name", type=str, default="our_svd_merging", help="name of the method to merge models",
                    choices=["ties_merging", "task_arithmetic", "fisher_merging", "regmean_merging", "our_merging", \
                             "ties_emr_merging", "t_switch_merging", "t_switch_diff_merging", "t_switch_svd_merging", \
                                "t_switch_module_merging", "our_svd_mask_merging", "personalized_mlp"])
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--gpu", type=int, default=6, help="number of gpu to use")
parser.add_argument("--top_k_ratio", type=float, default=1, help='Top K ratio for parameter selection')



try:
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    # args.device = f"cuda:{args.gpu}"
except:
    parser.print_help()
    sys.exit()

def state_dict_to_vector(state_dict, remove_keys=[], sort_keys=False):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(shared_state_dict.items())
    if sort_keys:
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
        ), list(sorted_shared_state_dict.keys())
    else:
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
        )

def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(reference_dict.items())

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict

def our_svd_mask_2(
    tensor: torch.Tensor, 
    density: float,
    **kwargs,
):
    if density > 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)
    if tensor.ndimension() == 0 :
        return tensor, tensor.numel() * 32
    
    if tensor.ndimension() == 1:
        tensor_flat = tensor
        tensor_mask_p_flat = torch.where(tensor_flat > 0, torch.tensor(1.0, device=tensor_flat.device), torch.tensor(-1.0, device=tensor_flat.device))
        tensor_mask_m_flat = torch.zeros_like(tensor_flat, dtype=torch.bool) 
        tensor_tv_scales_flat = torch.zeros_like(tensor_flat, dtype=torch.float32) 
        pos_idx = (tensor_flat > 0).nonzero(as_tuple=True)[0]  # 1D tensor of positions
        neg_idx = (tensor_flat < 0).nonzero(as_tuple=True)[0]

        pos_vals = tensor_flat[pos_idx]               # (num_pos,)
        neg_vals = tensor_flat[neg_idx]  

        num_pos = pos_vals.numel()
        num_neg = neg_vals.numel()

        k1_pos = int(num_pos * 0.5)
        k2_pos = int(num_pos * 1.0)
        k1_neg = int(num_neg * 0.5)
        k2_neg = int(num_neg * 1.0)

        if num_pos > 0:
            sorted_pos_rel = torch.argsort(pos_vals, descending=True)
            sorted_pos = pos_idx[sorted_pos_rel]

        if num_neg > 0:
            sorted_neg_rel = torch.argsort(neg_vals, descending=False)
            sorted_neg = neg_idx[sorted_neg_rel]

        if k1_pos > 0:
            tensor_mask_m_flat[sorted_pos[:k1_pos]] = True
            tensor_tv_scales_flat[sorted_pos[:k1_pos]] = torch.norm(tensor_flat[sorted_pos[:k1_pos]] *  tensor_mask_m_flat[sorted_pos[:k1_pos]], p=2) / torch.norm(tensor_mask_m_flat[sorted_pos[:k1_pos]]* tensor_mask_p_flat[sorted_pos[:k1_pos]], p=2)

        if k2_pos > k1_pos:
            tensor_mask_m_flat[sorted_pos[k1_pos:k2_pos]] = True
            tensor_tv_scales_flat[sorted_pos[k1_pos:k2_pos]] = torch.norm(tensor_flat[sorted_pos[k1_pos:k2_pos]] *  tensor_mask_m_flat[sorted_pos[k1_pos:k2_pos]], p=2) / torch.norm(tensor_mask_m_flat[sorted_pos[k1_pos:k2_pos]]* tensor_mask_p_flat[sorted_pos[k1_pos:k2_pos]], p=2)

        if k1_neg > 0:
            tensor_mask_m_flat[sorted_neg[:k1_neg]] = True
            tensor_tv_scales_flat[sorted_neg[:k1_neg]] = torch.norm(tensor_flat[sorted_neg[:k1_neg]] *  tensor_mask_m_flat[sorted_neg[:k1_neg]], p=2) / torch.norm(tensor_mask_m_flat[sorted_neg[:k1_neg]]* tensor_mask_p_flat[sorted_neg[:k1_neg]], p=2)
        if k2_neg > k1_neg:
            tensor_mask_m_flat[sorted_neg[k1_neg:k2_neg]] = True
            tensor_tv_scales_flat[sorted_neg[k1_neg:k2_neg]] = torch.norm(tensor_flat[sorted_neg[k1_neg:k2_neg]] *  tensor_mask_m_flat[sorted_neg[k1_neg:k2_neg]], p=2) / torch.norm(tensor_mask_m_flat[sorted_neg[k1_neg:k2_neg]]* tensor_mask_p_flat[sorted_neg[k1_neg:k2_neg]], p=2)

        tensor_ones = torch.ones_like(tensor)

        return tensor_mask_m_flat * tensor_mask_p_flat * tensor_tv_scales_flat * tensor_ones, tensor.numel() * 3
    
    driver = None
    if tensor.is_cuda:
        driver = 'gesvda'
    
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=True, driver=driver)
    new_rank = int(density * len(S))
    U, S, Vh = U[:, :new_rank], S[:new_rank], Vh[:new_rank, :]

    H, W = U.shape
    U_flat = U.reshape(-1) 
    U_mask_p_flat = torch.where(U_flat > 0, torch.tensor(1.0, device=U_flat.device), torch.tensor(-1.0, device=U_flat.device))
    U_mask_m_flat = torch.zeros_like(U_flat, dtype=torch.bool) 
    U_tv_scales_flat = torch.zeros_like(U_flat, dtype=torch.float32) 
    pos_idx = (U_flat > 0).nonzero(as_tuple=True)[0]  # 1D tensor of positions
    neg_idx = (U_flat < 0).nonzero(as_tuple=True)[0]

    pos_vals = U_flat[pos_idx]               # (num_pos,)
    neg_vals = U_flat[neg_idx]  

    num_pos = pos_vals.numel()
    num_neg = neg_vals.numel()

    k1_pos = int(num_pos * 0.5)
    k2_pos = int(num_pos * 1.0)
    k1_neg = int(num_neg * 0.5)
    k2_neg = int(num_neg * 1.0)

    if num_pos > 0:
        sorted_pos_rel = torch.argsort(pos_vals, descending=True)
        sorted_pos = pos_idx[sorted_pos_rel]

    if num_neg > 0:
        sorted_neg_rel = torch.argsort(neg_vals, descending=False)
        sorted_neg = neg_idx[sorted_neg_rel]

    if k1_pos > 0:
        U_mask_m_flat[sorted_pos[:k1_pos]] = True
        U_tv_scales_flat[sorted_pos[:k1_pos]] = torch.norm(U_flat[sorted_pos[:k1_pos]] *  U_mask_m_flat[sorted_pos[:k1_pos]], p=2) / torch.norm(U_mask_m_flat[sorted_pos[:k1_pos]]* U_mask_p_flat[sorted_pos[:k1_pos]], p=2)

    if k2_pos > k1_pos:
        U_mask_m_flat[sorted_pos[k1_pos:k2_pos]] = True
        U_tv_scales_flat[sorted_pos[k1_pos:k2_pos]] = torch.norm(U_flat[sorted_pos[k1_pos:k2_pos]] *  U_mask_m_flat[sorted_pos[k1_pos:k2_pos]], p=2) / torch.norm(U_mask_m_flat[sorted_pos[k1_pos:k2_pos]]* U_mask_p_flat[sorted_pos[k1_pos:k2_pos]], p=2)

    if k1_neg > 0:
        U_mask_m_flat[sorted_neg[:k1_neg]] = True
        U_tv_scales_flat[sorted_neg[:k1_neg]] = torch.norm(U_flat[sorted_neg[:k1_neg]] *  U_mask_m_flat[sorted_neg[:k1_neg]], p=2) / torch.norm(U_mask_m_flat[sorted_neg[:k1_neg]]* U_mask_p_flat[sorted_neg[:k1_neg]], p=2)
    if k2_neg > k1_neg:
        U_mask_m_flat[sorted_neg[k1_neg:k2_neg]] = True
        U_tv_scales_flat[sorted_neg[k1_neg:k2_neg]] = torch.norm(U_flat[sorted_neg[k1_neg:k2_neg]] *  U_mask_m_flat[sorted_neg[k1_neg:k2_neg]], p=2) / torch.norm(U_mask_m_flat[sorted_neg[k1_neg:k2_neg]]* U_mask_p_flat[sorted_neg[k1_neg:k2_neg]], p=2)

    U_mask_m = U_mask_m_flat.view(H, W)
    U_mask_p = U_mask_p_flat.view(H, W)
    U_tv_scales = U_tv_scales_flat.view(H, W)

    H, W = Vh.shape
    Vh_flat = Vh.reshape(-1) 
    Vh_mask_p_flat = torch.where(Vh_flat > 0, torch.tensor(1.0, device=Vh_flat.device), torch.tensor(-1.0, device=Vh_flat.device))
    Vh_mask_m_flat = torch.zeros_like(Vh_flat, dtype=torch.bool) 
    Vh_tv_scales_flat = torch.zeros_like(Vh_flat, dtype=torch.float32) 
    pos_idx = (Vh_flat > 0).nonzero(as_tuple=True)[0]  # 1D tensor of positions
    neg_idx = (Vh_flat < 0).nonzero(as_tuple=True)[0]

    pos_vals = Vh_flat[pos_idx]               # (num_pos,)
    neg_vals = Vh_flat[neg_idx]  

    num_pos = pos_vals.numel()
    num_neg = neg_vals.numel()

    k1_pos = int(num_pos * 0.5)
    k2_pos = int(num_pos * 1)
    k1_neg = int(num_neg * 0.5)
    k2_neg = int(num_neg * 1)

    if num_pos > 0:
        sorted_pos_rel = torch.argsort(pos_vals, descending=True)
        sorted_pos = pos_idx[sorted_pos_rel]

    if num_neg > 0:
        sorted_neg_rel = torch.argsort(neg_vals, descending=False)
        sorted_neg = neg_idx[sorted_neg_rel]

    if k1_pos > 0:
        Vh_mask_m_flat[sorted_pos[:k1_pos]] = True
        Vh_tv_scales_flat[sorted_pos[:k1_pos]] = torch.norm(Vh_flat[sorted_pos[:k1_pos]] *  Vh_mask_m_flat[sorted_pos[:k1_pos]], p=2) / torch.norm(Vh_mask_m_flat[sorted_pos[:k1_pos]]* Vh_mask_p_flat[sorted_pos[:k1_pos]], p=2)

    if k2_pos > k1_pos:
        Vh_mask_m_flat[sorted_pos[k1_pos:k2_pos]] = True
        Vh_tv_scales_flat[sorted_pos[k1_pos:k2_pos]] = torch.norm(Vh_flat[sorted_pos[k1_pos:k2_pos]] *  Vh_mask_m_flat[sorted_pos[k1_pos:k2_pos]], p=2) / torch.norm(Vh_mask_m_flat[sorted_pos[k1_pos:k2_pos]]* Vh_mask_p_flat[sorted_pos[k1_pos:k2_pos]], p=2)

    if k1_neg > 0:
        Vh_mask_m_flat[sorted_neg[:k1_neg]] = True
        Vh_tv_scales_flat[sorted_neg[:k1_neg]] = torch.norm(Vh_flat[sorted_neg[:k1_neg]] *  Vh_mask_m_flat[sorted_neg[:k1_neg]], p=2) / torch.norm(Vh_mask_m_flat[sorted_neg[:k1_neg]]* Vh_mask_p_flat[sorted_neg[:k1_neg]], p=2)
    if k2_neg > k1_neg:
        Vh_mask_m_flat[sorted_neg[k1_neg:k2_neg]] = True
        Vh_tv_scales_flat[sorted_neg[k1_neg:k2_neg]] = torch.norm(Vh_flat[sorted_neg[k1_neg:k2_neg]] *  Vh_mask_m_flat[sorted_neg[k1_neg:k2_neg]], p=2) / torch.norm(Vh_mask_m_flat[sorted_neg[k1_neg:k2_neg]]* Vh_mask_p_flat[sorted_neg[k1_neg:k2_neg]], p=2)

    Vh_mask_m = Vh_mask_m_flat.view(H, W)
    Vh_mask_p = Vh_mask_p_flat.view(H, W)
    Vh_tv_scales = Vh_tv_scales_flat.view(H, W)
    U_tensor_ones = torch.ones_like(U)
    Vh_tensor_ones = torch.ones_like(Vh)

    res = (U_mask_m * U_mask_p * U_tv_scales * U_tensor_ones) @ torch.diag(S) @ (Vh_mask_m * Vh_mask_p * Vh_tv_scales * Vh_tensor_ones)


    return res, U.numel() * 3 + S.numel() * 32 + Vh.numel() * 3

def our_svd_mask_1(
    tensor: torch.Tensor, 
    density: float,
    **kwargs,
):
    if density >= 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)
    if tensor.ndimension() == 0 :
        return tensor, tensor.numel() * 32
    
    if tensor.ndimension() == 1:
        tensor_flat = tensor
        tensor_mask_p_flat = torch.where(tensor_flat > 0, torch.tensor(1.0, device=tensor_flat.device), torch.tensor(-1.0, device=tensor_flat.device))
        tensor_mask_m_flat = torch.zeros_like(tensor_flat, dtype=torch.bool) 
        tensor_tv_scales_flat = torch.zeros_like(tensor_flat, dtype=torch.float32) 
        pos_idx = (tensor_flat > 0).nonzero(as_tuple=True)[0]  # 1D tensor of positions
        neg_idx = (tensor_flat < 0).nonzero(as_tuple=True)[0]

        pos_vals = tensor_flat[pos_idx]               # (num_pos,)
        neg_vals = tensor_flat[neg_idx]  

        num_pos = pos_vals.numel()
        num_neg = neg_vals.numel()

        k2_pos = int(num_pos * 1)
        k2_neg = int(num_neg * 1)

        if num_pos > 0:
            sorted_pos_rel = torch.argsort(pos_vals, descending=True)
            sorted_pos = pos_idx[sorted_pos_rel]

        if num_neg > 0:
            sorted_neg_rel = torch.argsort(neg_vals, descending=False)
            sorted_neg = neg_idx[sorted_neg_rel]

        if k2_pos > 0:
            tensor_mask_m_flat[sorted_pos[:k2_pos]] = True
            tensor_tv_scales_flat[sorted_pos[:k2_pos]] = torch.norm(tensor_flat[sorted_pos[:k2_pos]] *  tensor_mask_m_flat[sorted_pos[:k2_pos]], p=2) / torch.norm(tensor_mask_m_flat[sorted_pos[:k2_pos]]* tensor_mask_p_flat[sorted_pos[:k2_pos]], p=2)

        if k2_neg > 0:
            tensor_mask_m_flat[sorted_neg[:k2_neg]] = True
            tensor_tv_scales_flat[sorted_neg[:k2_neg]] = torch.norm(tensor_flat[sorted_neg[:k2_neg]] *  tensor_mask_m_flat[sorted_neg[:k2_neg]], p=2) / torch.norm(tensor_mask_m_flat[sorted_neg[:k2_neg]]* tensor_mask_p_flat[sorted_neg[:k2_neg]], p=2)

        tensor_ones = torch.ones_like(tensor)

        return tensor_mask_m_flat * tensor_mask_p_flat * tensor_tv_scales_flat * tensor_ones, tensor.numel() * 2
        return tensor, tensor.numel() * 2
    driver = None
    if tensor.is_cuda:
        driver = 'gesvda'
    
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=True, driver=driver)
    new_rank = int(density * len(S))
    U, S, Vh = U[:, :new_rank], S[:new_rank], Vh[:new_rank, :]

    H, W = U.shape
    U_flat = U.reshape(-1) 
    U_mask_p_flat = torch.where(U_flat > 0, torch.tensor(1.0, device=U_flat.device), torch.tensor(-1.0, device=U_flat.device))
    U_mask_m_flat = torch.zeros_like(U_flat, dtype=torch.bool) 
    U_tv_scales_flat = torch.zeros_like(U_flat, dtype=torch.float32) 
    pos_idx = (U_flat > 0).nonzero(as_tuple=True)[0]  # 1D tensor of positions
    neg_idx = (U_flat < 0).nonzero(as_tuple=True)[0] 

    pos_vals = U_flat[pos_idx]               # (num_pos,)
    neg_vals = U_flat[neg_idx]  

    num_pos = pos_vals.numel()
    num_neg = neg_vals.numel()

    k2_pos = int(num_pos * 1)
    k2_neg = int(num_neg * 1)

    if num_pos > 0:
        sorted_pos_rel = torch.argsort(pos_vals, descending=True)
        sorted_pos = pos_idx[sorted_pos_rel]

    if num_neg > 0:
        sorted_neg_rel = torch.argsort(neg_vals, descending=False)
        sorted_neg = neg_idx[sorted_neg_rel]


    if k2_pos > 0:
        U_mask_m_flat[sorted_pos[:k2_pos]] = True
        U_tv_scales_flat[sorted_pos[:k2_pos]] = torch.norm(U_flat[sorted_pos[:k2_pos]] *  U_mask_m_flat[sorted_pos[:k2_pos]], p=2) / torch.norm(U_mask_m_flat[sorted_pos[:k2_pos]]* U_mask_p_flat[sorted_pos[:k2_pos]], p=2)

    if k2_neg > 0:
        U_mask_m_flat[sorted_neg[:k2_neg]] = True
        U_tv_scales_flat[sorted_neg[:k2_neg]] = torch.norm(U_flat[sorted_neg[:k2_neg]] *  U_mask_m_flat[sorted_neg[:k2_neg]], p=2) / torch.norm(U_mask_m_flat[sorted_neg[:k2_neg]]* U_mask_p_flat[sorted_neg[:k2_neg]], p=2)

    U_mask_m = U_mask_m_flat.view(H, W)
    U_mask_p = U_mask_p_flat.view(H, W)
    U_tv_scales = U_tv_scales_flat.view(H, W)

    H, W = Vh.shape
    Vh_flat = Vh.reshape(-1) 
    Vh_mask_p_flat = torch.where(Vh_flat > 0, torch.tensor(1.0, device=Vh_flat.device), torch.tensor(-1.0, device=Vh_flat.device))
    Vh_mask_m_flat = torch.zeros_like(Vh_flat, dtype=torch.bool) 
    Vh_tv_scales_flat = torch.zeros_like(Vh_flat, dtype=torch.float32) 
    pos_idx = (Vh_flat > 0).nonzero(as_tuple=True)[0] # 1D tensor of positions
    neg_idx = (Vh_flat < 0).nonzero(as_tuple=True)[0]

    pos_vals = Vh_flat[pos_idx]               # (num_pos,)
    neg_vals = Vh_flat[neg_idx]  

    num_pos = pos_vals.numel()
    num_neg = neg_vals.numel()

    k2_pos = int(num_pos * 1)
    k2_neg = int(num_neg * 1)

    if num_pos > 0:
        sorted_pos_rel = torch.argsort(pos_vals, descending=True)
        sorted_pos = pos_idx[sorted_pos_rel]

    if num_neg > 0:
        sorted_neg_rel = torch.argsort(neg_vals, descending=False)
        sorted_neg = neg_idx[sorted_neg_rel]

    if k2_pos > 0:
        Vh_mask_m_flat[sorted_pos[:k2_pos]] = True
        Vh_tv_scales_flat[sorted_pos[:k2_pos]] = torch.norm(Vh_flat[sorted_pos[:k2_pos]] *  Vh_mask_m_flat[sorted_pos[:k2_pos]], p=2) / torch.norm(Vh_mask_m_flat[sorted_pos[:k2_pos]]* Vh_mask_p_flat[sorted_pos[:k2_pos]], p=2)

    if k2_neg > 0:
        Vh_mask_m_flat[sorted_neg[:k2_neg]] = True
        Vh_tv_scales_flat[sorted_neg[:k2_neg]] = torch.norm(Vh_flat[sorted_neg[:k2_neg]] *  Vh_mask_m_flat[sorted_neg[:k2_neg]], p=2) / torch.norm(Vh_mask_m_flat[sorted_neg[:k2_neg]]* Vh_mask_p_flat[sorted_neg[:k2_neg]], p=2)

    Vh_mask_m = Vh_mask_m_flat.view(H, W)
    Vh_mask_p = Vh_mask_p_flat.view(H, W)
    Vh_tv_scales = Vh_tv_scales_flat.view(H, W)
    U_tensor_ones = torch.ones_like(U)
    Vh_tensor_ones = torch.ones_like(Vh)

    res = (U_mask_m * U_mask_p * U_tv_scales * U_tensor_ones) @ torch.diag(S) @ (Vh_mask_m * Vh_mask_p * Vh_tv_scales * Vh_tensor_ones)


    return res, U.numel() * 2 + S.numel() * 32 + Vh.numel() * 2

def our_mask_svd(
    tensor: torch.Tensor, 
    density: float,
    **kwargs,
):
    if density >= 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)
    if len(tensor.shape) == 1 or tensor.shape[0] == 1 or tensor.shape[1] == 1 or len(tensor.shape) == 3 or tensor.ndimension() == 0:
        # rank=1
        return tensor, tensor.numel()
    

    tensor = tensor.float()
    driver = None
    if tensor.is_cuda:
        driver = 'gesvda'
    
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=True, driver=driver)
    new_rank = int(density * len(S))
    U, S, Vh = U[:, :new_rank], S[:new_rank], Vh[:new_rank, :]
    U = (U.bool()).float()
    S = S.bool().float()
    Vh = Vh.bool().float()
    res = U @ torch.diag(S) @ Vh
    res = res.bool()
    return res.reshape(-1) , U.numel() + S.numel() + Vh.numel()

def our_svd(
    tensor: torch.Tensor, 
    density: float,
    **kwargs,
):
    if density >= 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)
    if len(tensor.shape) == 1 or tensor.shape[0] == 1 or tensor.shape[1] == 1 or len(tensor.shape) == 3 or tensor.ndimension() == 0:
        # rank=1
        return tensor, tensor.numel()
    

    driver = None
    if tensor.is_cuda:
        driver = 'gesvda'
    
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=True, driver=driver)
    new_rank = int(density * len(S))
    U, S, Vh = U[:, :new_rank], S[:new_rank], Vh[:new_rank, :]
    U
    res = U @ torch.diag(S) @ Vh

    return res, U.numel() + S.numel() + Vh.numel()


def emr_merge_flat(task_vectors_flat: torch.Tensor):


    sum_param = task_vectors_flat.mean(dim=0)  # (m,)


    flag = torch.where(sum_param > 0,
                       torch.ones_like(sum_param),
                       -torch.ones_like(sum_param))  # (m,)



    param_signed = task_vectors_flat * flag.unsqueeze(0)  # (n, m)
    mask = param_signed > 0                              # (n, m) bool


    param_abs_masked = task_vectors_flat.abs() * mask     # 自动 cast to float
    #    param_max[j] = max_i param_abs_masked[i,j]
    param_max, _ = param_abs_masked.max(dim=0)           # (m,)


    vector_unified = param_max * flag                    # (m,)


    scales = task_vectors_flat.abs().mean(dim=1)         # (n,)


    new_scales = (vector_unified.abs().unsqueeze(0) * mask.float()).mean(dim=1)  # (n,)

    # 8) 计算 rescalers
    rescalers = scales / new_scales                      # (n,)

    return vector_unified, mask, rescalers


def task_vector_param_dict_to_single_vector(task_vector):
    task_vector_param_dict = copy.deepcopy(task_vector)
    sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))
    return torch.nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()])


def get_merge_OOD_performance(args: argparse.Namespace, models_to_merge: list, models_to_merge_ID: list, models_to_merge_OOD: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)

    # merged_params = merging_method.task_arithmetic(merged_model=merged_model, models_to_merge=models_to_merge_ID, exclude_param_names_regex=[".*classifier.*"],
    #                                              scaling_coefficient=0.35)
    merged_params = merging_method.ties_merging(merged_model=merged_model,
                                                   models_to_merge=models_to_merge_ID,
                                                   exclude_param_names_regex=[".*classifier.*"], param_value_mask_rate=0.2,  scaling_coefficient=1.2, 
                                                   ) 
    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_params:
                param_value.data.copy_(merged_params[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics

def get_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to(args.device)
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)

    set_random_seed(seed=0)

    merged_params = merging_method.merging_models(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"],
                                                   trainers = trainers, 
                                                   )

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to(args.device)
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_params:
                param_value.data.copy_(merged_params[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics


def get_ties_emr_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)


    merged_params = merging_method.ties_merging(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"], param_value_mask_rate=0.2,  scaling_coefficient=1.0, 
                                                   ) 
    remove_keys = ['classifier.dense.weight', 'classifier.dense.bias' , 'classifier.out_proj.weight' , 'classifier.out_proj.bias']
    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(), remove_keys) for check in models_to_merge])
    flat_ptm, sorted_list = state_dict_to_vector(merged_model.state_dict(), remove_keys, sort_keys=True)
    flat_merged = state_dict_to_vector(merged_params, remove_keys)

    tv_flat_checks = flat_ft - flat_ptm

    merged_tv = flat_merged

    scales = torch.zeros(tv_flat_checks.shape[0])

    flat_ft_sign = torch.sign(flat_ft)  # 
    merged_tv_sign = torch.sign(merged_tv).unsqueeze(0)  # 

    mask = (flat_ft_sign == merged_tv_sign)
    for idx in range(len(scales)):
        scales[idx] = torch.sum(torch.abs(tv_flat_checks[idx])) / torch.sum(torch.abs(merged_tv * mask[idx])) 

    merged_model.to(args.device)
    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )

        task_vector_recon = flat_ptm + scales[idx] * merged_tv * mask[idx]

        merged_state_dict = vector_to_state_dict(task_vector_recon, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics



def get_our_module_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)


    merged_params = merging_method.ties_merging(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"], param_value_mask_rate=0.2,  scaling_coefficient=1.0, 
                                                   ) 
    remove_keys = ['classifier.dense.weight', 'classifier.dense.bias' , 'classifier.out_proj.weight' , 'classifier.out_proj.bias']
    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(), remove_keys) for check in models_to_merge])
    flat_ptm, sorted_list = state_dict_to_vector(merged_model.state_dict(), remove_keys, sort_keys=True)
    flat_merged = state_dict_to_vector(merged_params, remove_keys)

    tv_flat_checks = flat_ft - flat_ptm

    merged_tv = flat_merged

    diff_flat_checks = tv_flat_checks - merged_tv

    merged_model_dict = merged_model.state_dict()
    filtered_keys = [k for k in sorted_list if "classifier" not in k]

    total_neuron_num = 0
    neuron_num_per_layer = []
    column_num_per_layer = []
    for n in filtered_keys:
        if merged_model_dict[n].dim()==0:
            total_neuron_num +=1
            neuron_num_per_layer.append(1)
            column_num_per_layer.append(1)
        elif merged_model_dict[n].dim()==1:
            total_neuron_num += merged_model_dict[n].size(0)
            neuron_num_per_layer.append(merged_model_dict[n].size(0))
            column_num_per_layer.append(1)
        elif merged_model_dict[n].dim()==2:
            total_neuron_num += merged_model_dict[n].size(0)
            neuron_num_per_layer.append(merged_model_dict[n].size(0))
            column_num_per_layer.append(merged_model_dict[n].size(1))
        else:
            total_neuron_num += merged_model_dict[n].size(0)
            neuron_num_per_layer.append(merged_model_dict[n].size(0))
            column_num_per_layer.append(merged_model_dict[n][0].numel())

    n_models = len(models_to_merge)
    avg_abs_diffs = torch.zeros(n_models, total_neuron_num)

    offset_row = 0
    offset_total = 0
    for idx_key, k in enumerate(filtered_keys):  
        rows = neuron_num_per_layer[idx_key]
        cols = column_num_per_layer[idx_key]
        for idx_row in range(rows):
            for idx_task in range(n_models):

                diffs= diff_flat_checks[idx_task][offset_total: offset_total + cols].abs().mean() 
                avg_abs_diffs[idx_task, offset_row] = diffs
            offset_row += 1
            offset_total += cols


    mask = torch.zeros_like(avg_abs_diffs, dtype=torch.bool)

    selected_neuron = int(total_neuron_num * args.top_k_ratio)


    Total_memory_bytes = 0
    for i in range(n_models):
        if selected_neuron > 0:

            _, idxs = torch.topk(avg_abs_diffs[i], selected_neuron)
            mask[i, idxs] = True
        total_params = 0       
        total_bytes  = 0       
        offset_row = 0 
        offset_total = 0
        for idx_key, k in enumerate(sorted_list):
            rows = neuron_num_per_layer[idx_key]
            cols = column_num_per_layer[idx_key]
            for idx_row in range(rows):
                m = mask[i][offset_row]      
                if m: # shape=(rows,)
                    param = flat_ptm[offset_total: offset_total + cols]     # Tensor, shape=() 或 (rows, ...)
                    total_params += param.numel() 
                    total_bytes  += param.numel() * param.element_size() 
                offset_row += 1
                offset_total += cols


        Total_memory_bytes += total_bytes



    mask_merge = torch.zeros_like(diff_flat_checks, dtype=torch.bool)

    offset_row = 0
    offset_total = 0
    for idx_key, k in enumerate(sorted_list):  
        rows = neuron_num_per_layer[idx_key]
        cols = column_num_per_layer[idx_key]
        for idx_row in range(rows):
            for idx_task in range(n_models):
                m = mask[idx_task][offset_row]   
                mask_merge[idx_task][offset_total: offset_total + cols] = m
            offset_row += 1
            offset_total += cols

    merged_model.to(args.device)
    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )

        task_vector_recon = flat_merged * (~mask_merge[idx]) + flat_ft[idx] * mask_merge[idx]

        merged_state_dict = vector_to_state_dict(task_vector_recon, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics



def get_our_align_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to(args.device)
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)

    set_random_seed(seed=0)

    merged_params = merging_method.merging_models(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"]
                                                   ) 

    remove_keys = ['classifier.dense.weight', 'classifier.dense.bias' , 'classifier.out_proj.weight' , 'classifier.out_proj.bias']
    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(), remove_keys) for check in models_to_merge])
    flat_ptm, sorted_list = state_dict_to_vector(merged_model.state_dict(), remove_keys, sort_keys=True)
    flat_merged = state_dict_to_vector(merged_params, remove_keys)

    tv_flat_checks = flat_ft - flat_ptm

    merged_tv = torch.sum(tv_flat_checks, dim=0) 

    diff_flat_checks = tv_flat_checks - merged_tv

    merged_model_dict = merged_model.state_dict()
    filtered_keys = [k for k in sorted_list if "classifier" not in k]

    total_neuron_num = 0
    neuron_num_per_layer = []
    column_num_per_layer = []
    for n in filtered_keys:
        if 'bias' in n :
            continue
        elif merged_model_dict[n].dim()==0:
            total_neuron_num +=1
            neuron_num_per_layer.append(1)
            column_num_per_layer.append(1)
        elif merged_model_dict[n].dim()==1:
            total_neuron_num += merged_model_dict[n].size(0)
            neuron_num_per_layer.append(merged_model_dict[n].size(0))
            column_num_per_layer.append(1)
        elif merged_model_dict[n].dim()==2:
            total_neuron_num += merged_model_dict[n].size(0)
            neuron_num_per_layer.append(merged_model_dict[n].size(0))
            column_num_per_layer.append(merged_model_dict[n].size(1))
        else:
            total_neuron_num += merged_model_dict[n].size(0)
            neuron_num_per_layer.append(merged_model_dict[n].size(0))
            column_num_per_layer.append(merged_model_dict[n][0].numel())

    n_models = len(models_to_merge)
    avg_abs_diffs = torch.zeros(n_models, total_neuron_num)

    offset_row = 0
    offset_total = 0
    layer_num_location = 0
    for idx_key, key in enumerate(filtered_keys):  
        if 'bias' in key :
            continue
        elif 'bias' in filtered_keys[idx_key+1]:
            rows = neuron_num_per_layer[layer_num_location]
            cols = column_num_per_layer[layer_num_location]
            bias_location = offset_total + rows * cols
            for idx_row in range(rows):
                for idx_task in range(n_models):
                    single_neuron = torch.cat((diff_flat_checks[idx_task][offset_total: offset_total + cols], diff_flat_checks[idx_task][bias_location].unsqueeze(0)))

                    diffs = torch.mean(torch.abs(single_neuron)) 
                    avg_abs_diffs[idx_task, offset_row] = diffs
                offset_row += 1
                offset_total += cols
                bias_location += 1
            offset_total += rows
            layer_num_location += 1
        else:
            rows = neuron_num_per_layer[layer_num_location]
            cols = column_num_per_layer[layer_num_location]
            for idx_row in range(rows):
                for idx_task in range(n_models):
                    single_neuron = diff_flat_checks[idx_task][offset_total: offset_total + cols]

                    diffs = torch.mean(torch.abs(single_neuron)) 
                    avg_abs_diffs[idx_task, offset_row] = diffs
                offset_row += 1
                offset_total += cols
            layer_num_location += 1

    mask = torch.zeros_like(avg_abs_diffs, dtype=torch.bool)

    selected_neuron = int(total_neuron_num * args.top_k_ratio)


    Total_memory_bytes = 0
    for i in range(n_models):
        if selected_neuron > 0:

            _, idxs = torch.topk(avg_abs_diffs[i], selected_neuron)
            mask[i, idxs] = True
        total_params = 0      
        total_bytes  = 0       
        offset_row = 0
        offset_total = 0
        layer_num_location = 0
        for idx_key, key in enumerate(filtered_keys):
            if 'bias' in key :
                continue
            elif 'bias' in filtered_keys[idx_key+1]:
                rows = neuron_num_per_layer[layer_num_location]
                cols = column_num_per_layer[layer_num_location]
                for idx_row in range(rows):
                    m = mask[i][offset_row]      
                    if m: # shape=(rows,)
                        param = flat_ptm[offset_total: offset_total + cols]  
                        total_params += (param.numel() + 1)
                        total_bytes  += (param.numel() + 1) * param.element_size() 
                    offset_row += 1
                    offset_total += cols
                offset_total += rows
                layer_num_location += 1
            else:
                rows = neuron_num_per_layer[layer_num_location]
                cols = column_num_per_layer[layer_num_location]
                for idx_row in range(rows):
                    m = mask[i][offset_row]      
                    if m: # shape=(rows,)
                        param = flat_ptm[offset_total: offset_total + cols]    
                        total_params += (param.numel())
                        total_bytes  += (param.numel()) * param.element_size() 
                    offset_row += 1
                    offset_total += cols    
                layer_num_location += 1


        Total_memory_bytes += total_bytes



    mask_merge = torch.zeros_like(diff_flat_checks, dtype=torch.bool)

    offset_row = 0
    offset_total = 0
    layer_num_location = 0
    for idx_key, key in enumerate(filtered_keys):  
        if 'bias' in key :
            continue
        elif 'bias' in filtered_keys[idx_key+1]:
            rows = neuron_num_per_layer[layer_num_location]
            cols = column_num_per_layer[layer_num_location]
            bias_location = offset_total + rows * cols
            for idx_row in range(rows):
                for idx_task in range(n_models):
                    m = mask[idx_task][offset_row]  
                    mask_merge[idx_task][offset_total: offset_total + cols] = m
                    mask_merge[idx_task][bias_location] = m
                offset_row += 1
                offset_total += cols
                bias_location += 1
            offset_total += rows
            layer_num_location += 1
        else:
            rows = neuron_num_per_layer[layer_num_location]
            cols = column_num_per_layer[layer_num_location]
            for idx_row in range(rows):
                for idx_task in range(n_models):
                    m = mask[idx_task][offset_row]   
                    mask_merge[idx_task][offset_total: offset_total + cols] = m
                offset_row += 1
                offset_total += cols
            layer_num_location += 1


    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )

        task_vector_recon = flat_merged * (~mask_merge[idx]) + flat_ft[idx] * mask_merge[idx]

        merged_state_dict = vector_to_state_dict(task_vector_recon, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics


def get_our_svd_mask_merge_OOD_performance(args: argparse.Namespace, models_to_merge: list,models_to_merge_ID: list,models_to_merge_OOD: list, trainers: list, trainers_OOD: list, trainers_ID: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)



    merged_parameter = merging_method.ties_merging(merged_model=merged_model, models_to_merge=models_to_merge_ID, exclude_param_names_regex=[".*classifier.*"],
                                                 scaling_coefficient=1.0)
    remove_keys = ['out_proj.weight', 'out_proj.bias']
    for idx_dataname, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names_OOD, models_to_merge_OOD, trainers_OOD)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        
        head_cos_smi= torch.zeros(len(models_to_merge_ID))
        classification_head_this_dataset = state_dict_to_vector(model_to_merge.classifier.state_dict(), remove_keys = remove_keys)
        for idx_dataname, model_to_merge_ID in enumerate(models_to_merge_ID):
            classification_head = state_dict_to_vector(model_to_merge_ID.classifier.state_dict(), remove_keys = remove_keys)
            cos_sim = F.cosine_similarity(classification_head_this_dataset, classification_head, dim=0)
            head_cos_smi[idx_dataname] = cos_sim
        head_cos_smi = head_cos_smi / head_cos_smi.sum()
        for idx, (param_name, param_value) in enumerate(merged_model.named_parameters()):
            if param_name in merged_parameter:
                svd_param_temp = torch.zeros_like(merged_parameter[param_name])
                for idx_dataname, model_to_merge_ID in enumerate(models_to_merge_ID):
                    svd_param, total_parameter_num_svd = our_svd_mask_2(model_to_merge_ID.state_dict()[param_name] - merged_parameter[param_name], 0.1)
                    svd_param_temp += svd_param * head_cos_smi[idx_dataname]
                param_value.data.copy_(svd_param_temp + merged_parameter[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training argument
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

   

    return test_metrics




def get_our_svd_mask_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)


    merged_parameter = merging_method.ties_merging(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=[".*classifier.*"],
                                                 scaling_coefficient=0.3)
    
    models_to_merge_state = [model.state_dict() for model in models_to_merge]
    pretrain_model_state = merged_model.state_dict()
    mean_parameter_memory = 0
    mean_parameter_memory_svd = 0
    for idx_dataname, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        total_parameter_nums = 0
        total_parameter_nums_svd = 0
        for idx, (param_name, param_value) in enumerate(merged_model.named_parameters()):
            if param_name in merged_parameter:
                # svd_param, total_parameter_num_svd = our_svd_mask_2(models_to_merge_state[idx_dataname][param_name] - merged_parameter[param_name], 0.08)
                # total_parameter_nums += merged_parameter[param_name].numel()
                # total_parameter_nums_svd += total_parameter_num_svd
                # param_value.data.copy_(svd_param + merged_parameter[param_name])
                svd_param, total_parameter_num_svd = our_svd_mask_2(models_to_merge_state[idx_dataname][param_name] - pretrain_model_state[param_name], 0.1)
                total_parameter_nums += pretrain_model_state[param_name].numel()
                total_parameter_nums_svd += total_parameter_num_svd
                param_value.data.copy_(svd_param + pretrain_model_state[param_name])

        

        mean_parameter_memory += total_parameter_nums/1024/1024 * 4
        mean_parameter_memory_svd += total_parameter_nums_svd/1024/1024 /8 

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training argument
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")



    total_size = 0
    for name, param in merged_model.named_parameters():
        total_size += param.numel() * param.element_size() 


    return test_metrics


def get_our_svd_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)

    merged_parameter = merging_method.task_arithmetic(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=[".*classifier.*"],
                                                 scaling_coefficient=0.3)

    pretrain_model_state = merged_model.state_dict()
    models_to_merge_state = [model.state_dict() for model in models_to_merge]

    mean_parameter_memory = 0
    mean_parameter_memory_svd = 0
    for idx_dataname, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",  # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )

        total_parameter_nums = 0
        total_parameter_nums_svd = 0
        for idx, (param_name, param_value) in enumerate(merged_model.named_parameters()):
            if param_name in merged_parameter:
                svd_param, total_parameter_num_svd = our_svd(models_to_merge_state[idx_dataname][param_name] - pretrain_model_state[param_name], 0.03)
                total_parameter_nums += pretrain_model_state[param_name].numel()
                total_parameter_nums_svd += total_parameter_num_svd
                param_value.data.copy_(svd_param + pretrain_model_state[param_name])


        mean_parameter_memory += total_parameter_nums/1024/1024 * 4
        mean_parameter_memory_svd += total_parameter_nums_svd/1024/1024 * 4

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  
            args=merged_model_training_args,  
            eval_dataset=trainer.eval_dataset,  
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  
            tokenizer=tokenizer  
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")



    total_size = 0
    for name, param in merged_model.named_parameters():
        total_size += param.numel() * param.element_size() 



    return test_metrics


def get_our_t_switch_module_merge_performance_beifen(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)

    # merged_task_vector, models_to_merge_task_vectors = merging_method.our_task_arithmetic(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=[".*classifier.*"],
    #                                              scaling_coefficient=0.3)
    alpha = 0.35

    merged_parameter = merging_method.task_arithmetic(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=[".*classifier.*"],
                                                 scaling_coefficient=0.3)
       
    # index_to_key = {index: key for index, key in enumerate(merged_params.keys())}
    n_models = len(models_to_merge)
    total_layer = len(merged_parameter)
    diff_t_merge = torch.zeros(n_models, total_layer)
    pretrain_model_state = merged_model.state_dict()
    models_to_merge_state = [model.state_dict() for model in models_to_merge]
    for model_num in range(len(models_to_merge)):
        for idx_key, key in enumerate(merged_parameter): 
            diff_t_merge[model_num][idx_key] = torch.mean(torch.abs(models_to_merge_state[model_num][key] - merged_parameter[key]))

    selected_layer = int(total_layer * args.top_k_ratio)

    mean_parameter_memory = 0
    for idx_dataname, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        _, selected_idxs = torch.topk(diff_t_merge[idx_dataname], selected_layer)
        total_parameter_nums = 0
        for idx, (param_name, param_value) in enumerate(merged_model.named_parameters()):
            if param_name in merged_parameter:
                if idx in selected_idxs:
                    para = models_to_merge_state[idx_dataname][param_name] - merged_parameter[param_name]
                    flat_para = para.reshape(-1)
                    mask_m = torch.zeros_like(flat_para, dtype=torch.bool)
                    mask_p = torch.where(flat_para > 0, torch.tensor(1.0, device=flat_para.device), torch.tensor(-1.0, device=flat_para.device))
                    pos_mask = flat_para > 0
                    pos_values = flat_para[pos_mask]
                    if pos_values.numel() > 0:
                        k_pos = int(pos_values.numel() * alpha)
                        if k_pos > 0:
                            topk_pos_values, topk_pos_indices = torch.topk(pos_values, k=k_pos)
                            pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)
                            selected_pos_indices = pos_indices[topk_pos_indices]
                            mask_m[selected_pos_indices] = True

                    neg_mask = flat_para < 0
                    neg_values = flat_para[neg_mask]
                    if neg_values.numel() > 0:
                        k_neg = int(neg_values.numel() * alpha)
                        if k_neg > 0:

                            bottomk_neg_values, bottomk_neg_indices = torch.topk(neg_values, k=k_neg, largest=False)
                            neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(1)
                            selected_neg_indices = neg_indices[bottomk_neg_indices]
                            mask_m[selected_neg_indices] = True

                    tensor_ones = torch.ones_like(flat_para)
                    tv_scales = torch.norm(flat_para *  mask_m, p=2) / torch.norm(mask_m* mask_p, p=2)

                    switch_para = tensor_ones * tv_scales * mask_m * mask_p
                    switch_para_state = switch_para.reshape(para.shape)
                    total_parameter_nums += len(flat_para) * 2
                    
                    param_value.data.copy_(switch_para_state + merged_parameter[param_name])
                    # svd_param, total_parameter_num_svd = our_svd(models_to_merge_state[idx_dataname][param_name] - pretrain_model_state[param_name], 0.04)
                    # total_parameter_nums += pretrain_model_state[param_name].numel()
                    # total_parameter_nums_svd += total_parameter_num_svd
                    # param_value.data.copy_(svd_param + pretrain_model_state[param_name])
                else:
                    param_value.data.copy_(merged_parameter[param_name])
        


        mean_parameter_memory += total_parameter_nums/1024/1024/8

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training argument
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")


    total_size = 0
    for name, param in merged_model.named_parameters():
        total_size += param.numel() * param.element_size() 



    return test_metrics


def get_our_parameter_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)



    merged_params = merging_method.task_arithmetic(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"],  scaling_coefficient=0.3, 
                                                   ) 

    remove_keys = ['classifier.dense.weight', 'classifier.dense.bias' , 'classifier.out_proj.weight' , 'classifier.out_proj.bias']
    print(f"Flattening out Checkpoints")

    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(), remove_keys) for check in models_to_merge])
    flat_merged = state_dict_to_vector(merged_params, remove_keys)

    diff_flat_checks = flat_ft - flat_merged
    mask = torch.zeros_like(diff_flat_checks, dtype=torch.bool)
    num_tasks, num_params = diff_flat_checks.shape
    for i in range(num_tasks):

        abs_values = torch.abs(diff_flat_checks[i])


        k = int(num_params * top_k_ratio)
        if k == 0:
            continue
        topk_vals, topk_indices = torch.topk(abs_values, k)

  
        mask[i, topk_indices] = True

    merged_model.to(args.device)
    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )

        task_vector_recon = flat_merged * (~mask[idx]) + flat_ft[idx] * mask[idx]

        merged_state_dict = vector_to_state_dict(task_vector_recon, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics

# ************************** T-switch ******************************
def get_t_switch_diff_vector_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    # ****************** top alpha ************************
    alpha = 0.35

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)

    scaling_coef_ = 0.3

    remove_keys = ['classifier.dense.weight', 'classifier.dense.bias' , 'classifier.out_proj.weight' , 'classifier.out_proj.bias']
    print(f"Flattening out Checkpoints")

    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(), remove_keys) for check in models_to_merge])
    flat_ptm = state_dict_to_vector(merged_model.state_dict(), remove_keys)
    tv_flat_checks = flat_ft - flat_ptm
    task_vector_sum = sum(tv_flat_checks)
    merged_task_vector = scaling_coef_ * task_vector_sum
    flat_merged_mdoel = flat_ptm + merged_task_vector

    diff_vector_flat_checks = flat_ft - flat_merged_mdoel

    mask_m = torch.zeros_like(diff_vector_flat_checks, dtype=torch.bool)
    for idx in range(diff_vector_flat_checks.size(0)):        
        pos_mask = diff_vector_flat_checks[idx] > 0
        pos_values = diff_vector_flat_checks[idx][pos_mask]
        if pos_values.numel() > 0:
            k_pos = int(pos_values.numel() * alpha)
            if k_pos > 0:

                topk_pos_values, topk_pos_indices = torch.topk(pos_values, k=k_pos)
                pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)
                selected_pos_indices = pos_indices[topk_pos_indices]
                mask_m[idx][selected_pos_indices] = True


        neg_mask = diff_vector_flat_checks[idx]  < 0
        neg_values = diff_vector_flat_checks[idx][neg_mask]
        if neg_values.numel() > 0:
            k_neg = int(neg_values.numel() * alpha)
            if k_neg > 0:

                bottomk_neg_values, bottomk_neg_indices = torch.topk(neg_values, k=k_neg, largest=False)
                neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(1)
                selected_neg_indices = neg_indices[bottomk_neg_indices]
                mask_m[idx][selected_neg_indices] = True



    mask_p = torch.where(diff_vector_flat_checks > 0, torch.tensor(1.0, device=diff_vector_flat_checks.device), torch.tensor(-1.0, device=diff_vector_flat_checks.device))

    tensor_ones = torch.ones_like(diff_vector_flat_checks[0])

    tv_scales = torch.zeros(diff_vector_flat_checks.shape[0])
    for idx in range(diff_vector_flat_checks.shape[0]):
        tv_scales[idx] = torch.norm((diff_vector_flat_checks[idx]) *  mask_m[idx], p=2) / torch.norm(mask_m[idx]* mask_p[idx], p=2)


    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )

        model_vector_recon = flat_merged_mdoel + tv_scales[idx] * mask_m[idx] * mask_p[idx] * tensor_ones

        merged_state_dict = vector_to_state_dict(model_vector_recon, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics

# ************************** T-switch-mask ******************************
def get_t_switch_mask_vector_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    # ****************** top alpha ************************
    alpha = 0.35

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)


    remove_keys = ['classifier.dense.weight', 'classifier.dense.bias' , 'classifier.out_proj.weight' , 'classifier.out_proj.bias']
    print(f"Flattening out Checkpoints")

    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(), remove_keys) for check in models_to_merge])
    flat_ptm = state_dict_to_vector(merged_model.state_dict(), remove_keys)
    tv_flat_checks = flat_ft - flat_ptm


    mask_m = torch.zeros_like(tv_flat_checks, dtype=torch.bool)
    for idx in range(tv_flat_checks.size(0)):        
        pos_mask = tv_flat_checks[idx] > 0
        pos_values = tv_flat_checks[idx][pos_mask]
        if pos_values.numel() > 0:
            k_pos = int(pos_values.numel() * alpha)
            if k_pos > 0:

                topk_pos_values, topk_pos_indices = torch.topk(pos_values, k=k_pos)
                pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)
                selected_pos_indices = pos_indices[topk_pos_indices]
                mask_m[idx][selected_pos_indices] = True


        neg_mask = tv_flat_checks[idx]  < 0
        neg_values = tv_flat_checks[idx][neg_mask]
        if neg_values.numel() > 0:
            k_neg = int(neg_values.numel() * alpha)
            if k_neg > 0:

                bottomk_neg_values, bottomk_neg_indices = torch.topk(neg_values, k=k_neg, largest=False)
                neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(1)
                selected_neg_indices = neg_indices[bottomk_neg_indices]
                mask_m[idx][selected_neg_indices] = True


    mask_p = torch.where(tv_flat_checks > 0, torch.tensor(1.0, device=tv_flat_checks.device), torch.tensor(-1.0, device=tv_flat_checks.device))

    tensor_ones = torch.ones_like(tv_flat_checks[0])

    tv_scales = torch.zeros(tv_flat_checks.shape[0])
    for idx in range(tv_flat_checks.shape[0]):
        tv_scales[idx] = torch.norm((tv_flat_checks[idx]) *  mask_m[idx], p=2) / torch.norm(mask_m[idx]* mask_p[idx], p=2)

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )

        model_vector_recon = flat_ptm + tv_scales[idx] * mask_m[idx] * mask_p[idx] * tensor_ones

        merged_state_dict = vector_to_state_dict(model_vector_recon, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics

def get_t_switch_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    # ****************** top alpha ************************
    alpha = 0.35

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)

    remove_keys = ['classifier.dense.weight', 'classifier.dense.bias' , 'classifier.out_proj.weight' , 'classifier.out_proj.bias']
    print(f"Flattening out Checkpoints")

    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(), remove_keys) for check in models_to_merge])
    flat_ptm = state_dict_to_vector(merged_model.state_dict(), remove_keys)
    flat_task_vector = torch.sum(flat_ft - flat_ptm, dim=0) * 0.3
    flat_merged = flat_ptm + flat_task_vector
    tv_flat_checks = flat_ft - flat_ptm
    diff_flat_checks = flat_ft - flat_merged

    mask_p = torch.where(diff_flat_checks > 0, torch.tensor(1.0, device=diff_flat_checks.device), torch.tensor(-1.0, device=diff_flat_checks.device))
    mask_m = torch.zeros_like(diff_flat_checks, dtype=torch.bool)
    for idx in range(diff_flat_checks.size(0)):        
        pos_mask = diff_flat_checks[idx] > 0
        pos_values = diff_flat_checks[idx][pos_mask]
        if pos_values.numel() > 0:
            k_pos = int(pos_values.numel() * alpha)
            if k_pos > 0:
                topk_pos_values, topk_pos_indices = torch.topk(pos_values, k=k_pos)
                pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)
                selected_pos_indices = pos_indices[topk_pos_indices]
                mask_m[idx][selected_pos_indices] = True

        neg_mask = diff_flat_checks[idx]  < 0
        neg_values = diff_flat_checks[idx][neg_mask]
        if neg_values.numel() > 0:
            k_neg = int(neg_values.numel() * alpha)
            if k_neg > 0:
 
                bottomk_neg_values, bottomk_neg_indices = torch.topk(neg_values, k=k_neg, largest=False)
                neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(1)
                selected_neg_indices = neg_indices[bottomk_neg_indices]
                mask_m[idx][selected_neg_indices] = True

    tensor_ones = torch.ones_like(diff_flat_checks[0])

    tv_scales = torch.zeros(diff_flat_checks.shape[0])
    for idx in range(diff_flat_checks.shape[0]):
        tv_scales[idx] = torch.norm(diff_flat_checks[idx] *  mask_m[idx], p=2) / torch.norm(mask_m[idx]* mask_p[idx], p=2)

    task_vector_recon = torch.ones_like(diff_flat_checks)
    for idx in range(diff_flat_checks.shape[0]):
        task_vector_recon[idx] = tv_scales[idx] * mask_m[idx] * mask_p[idx] * tensor_ones

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )

        model_vector_recon = flat_merged + task_vector_recon[idx]
        # model_vector_recon = flat_merged
        merged_state_dict = vector_to_state_dict(model_vector_recon, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics


def get_selected_parameter(task_m: torch.Tensor, model_state,  sorted_list, thre):
    total_para_num = 0
    offset_total = 0
    task_new = torch.zeros_like(task_m)
    for module_name in model_state:
        if module_name in sorted_list:
            if model_state[module_name].ndimension() == 1:
                neuron_size = len(model_state[module_name])
                para_now  =  task_m[offset_total: offset_total + neuron_size]
                if torch.sum(para_now.float()) == 0:
                    task_new[offset_total: offset_total + neuron_size] = False
                else:
                    task_new[offset_total: offset_total + neuron_size] = para_now
                    total_para_num += neuron_size
                offset_total += neuron_size
            else:
                neuron_num = model_state[module_name].shape[0]
                neuron_size = model_state[module_name].shape[1]
                for idx in range(neuron_num):
                    para_now  =  task_m[offset_total: offset_total + neuron_size]
                    if torch.mean(para_now.float()) > thre:
                        total_para_num += neuron_size
                        task_new[offset_total: offset_total + neuron_size] = para_now
                    else:
                        task_new[offset_total: offset_total + neuron_size] = False
                    offset_total += neuron_size

  
    return task_new, total_para_num, offset_total

def get_half_parameter(task_m: torch.Tensor, model_state,  sorted_list, thre):
    offset_total = 0
    task_new = torch.zeros_like(task_m)
    for module_name in model_state:
        if module_name in sorted_list:
            if model_state[module_name].ndimension() == 1:
                neuron_size = len(model_state[module_name])
                para_now  =  task_m[offset_total: offset_total + neuron_size]
                if torch.sum(para_now.float()) == 0:
                    task_new[offset_total: offset_total + neuron_size] = 0
                elif torch.sum(para_now.float()) > 0:
                    task_new[offset_total: offset_total + neuron_size] = 1
                else:
                    task_new[offset_total: offset_total + neuron_size] = -1
                offset_total += neuron_size
            else:
                neuron_num = model_state[module_name].shape[0]
                neuron_size = model_state[module_name].shape[1]
                for idx in range(neuron_num):
                    para_now  =  task_m[offset_total: offset_total + neuron_size]
                    if torch.sum(para_now.float()) == 0:
                        task_new[offset_total: offset_total + neuron_size] = 0
                    elif torch.sum(para_now.float()) > 0:
                        task_new[offset_total: offset_total + neuron_size] = 1
                    else:
                        task_new[offset_total: offset_total + neuron_size] = -1
                    offset_total += neuron_size

   

    return task_new

def get_t_switch_multi_msak_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    # ****************** top alpha ************************
    alpha = 0.35

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='./datasets').to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)

    remove_keys = ['classifier.dense.weight', 'classifier.dense.bias' , 'classifier.out_proj.weight' , 'classifier.out_proj.bias']
    print(f"Flattening out Checkpoints")
    pretrain_state = merged_model.state_dict()
    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(), remove_keys) for check in models_to_merge])
    flat_ptm, selected_list = state_dict_to_vector(merged_model.state_dict(), remove_keys, sort_keys=True)
    tv_flat_checks = flat_ft - flat_ptm

    mask_p = torch.where(tv_flat_checks > 0, torch.tensor(1.0, device=tv_flat_checks.device), torch.tensor(-1.0, device=tv_flat_checks.device))
    tv_scales = torch.zeros_like(tv_flat_checks)

    mask_m = torch.zeros_like(tv_flat_checks, dtype=torch.bool)
    for idx in range(tv_flat_checks.size(0)):        
        pos_mask = tv_flat_checks[idx] > 0
        pos_values = tv_flat_checks[idx][pos_mask]
        if pos_values.numel() > 0:
            k_pos = int(pos_values.numel() * alpha)
            if k_pos > 0:
                sorted_values, sorted_indices = torch.sort(pos_values, descending=True)
                top_10_percent_idx = int(len(pos_values) * 0.15)
                # top_20_percent_idx = int(len(pos_values) * 0.25)
                top_30_percent_idx = int(len(pos_values) * 0.35)
                selected_indices_top = sorted_indices[: top_10_percent_idx]
                # selected_indices_middile = sorted_indices[top_10_percent_idx: top_20_percent_idx]
                selected_indices_bottom = sorted_indices[top_10_percent_idx: top_30_percent_idx]
                pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)
                selected_indices_top = pos_indices[selected_indices_top]
                # selected_indices_middile = pos_indices[selected_indices_middile]
                selected_indices_bottom = pos_indices[selected_indices_bottom]

                mask_m[idx][selected_indices_top] = True
                # mask_m[idx][selected_indices_middile] = True
                mask_m[idx][selected_indices_bottom] = True
                tv_scales[idx][selected_indices_top] = torch.norm((tv_flat_checks[idx][selected_indices_top]) *  mask_m[idx][selected_indices_top], p=2) / torch.norm(mask_m[idx][selected_indices_top] * mask_p[idx][selected_indices_top], p=2)
                # tv_scales[idx][selected_indices_middile] = torch.norm((tv_flat_checks[idx][selected_indices_middile]) *  mask_m[idx][selected_indices_middile], p=2) / torch.norm(mask_m[idx][selected_indices_middile] * mask_p[idx][selected_indices_middile], p=2)
                tv_scales[idx][selected_indices_bottom] = torch.norm((tv_flat_checks[idx][selected_indices_bottom]) *  mask_m[idx][selected_indices_bottom], p=2) / torch.norm(mask_m[idx][selected_indices_bottom] * mask_p[idx][selected_indices_bottom], p=2)

  
        neg_mask = tv_flat_checks[idx]  < 0
        neg_values = tv_flat_checks[idx][neg_mask]
        if neg_values.numel() > 0:
            k_neg = int(neg_values.numel() * alpha)
            if k_neg > 0:
                neg_sorted_values, neg_sorted_indices = torch.sort(neg_values, descending=False)
                neg_top_10_percent_idx = int(len(neg_values) * 0.15)
                # neg_top_20_percent_idx = int(len(neg_values) * 0.25)
                neg_top_30_percent_idx = int(len(neg_values) * 0.35)
                neg_selected_indices_top = neg_sorted_indices[: neg_top_10_percent_idx]
                # neg_selected_indices_middile = neg_sorted_indices[neg_top_10_percent_idx: neg_top_20_percent_idx]
                neg_selected_indices_bottom = neg_sorted_indices[neg_top_10_percent_idx: neg_top_30_percent_idx]
                neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(1)
                neg_selected_indices_top = neg_indices[neg_selected_indices_top]
                # neg_selected_indices_middile = neg_indices[neg_selected_indices_middile]
                neg_selected_indices_bottom = neg_indices[neg_selected_indices_bottom]
  
                mask_m[idx][neg_selected_indices_top] = True
                # mask_m[idx][neg_selected_indices_middile] = True
                mask_m[idx][neg_selected_indices_bottom] = True
                tv_scales[idx][neg_selected_indices_top] = torch.norm((tv_flat_checks[idx][neg_selected_indices_top]) *  mask_m[idx][neg_selected_indices_top], p=2) / torch.norm(mask_m[idx][neg_selected_indices_top] * mask_p[idx][neg_selected_indices_top], p=2)
                # tv_scales[idx][neg_selected_indices_middile] = torch.norm((tv_flat_checks[idx][neg_selected_indices_middile]) *  mask_m[idx][neg_selected_indices_middile], p=2) / torch.norm(mask_m[idx][neg_selected_indices_middile] * mask_p[idx][neg_selected_indices_middile], p=2)
                tv_scales[idx][neg_selected_indices_bottom] = torch.norm((tv_flat_checks[idx][neg_selected_indices_bottom]) *  mask_m[idx][neg_selected_indices_bottom], p=2) / torch.norm(mask_m[idx][neg_selected_indices_bottom] * mask_p[idx][neg_selected_indices_bottom], p=2)

    tensor_ones = torch.ones_like(tv_flat_checks[0])

    # task_vector_recon = torch.ones_like(tv_flat_checks)
    # for idx in range(tv_flat_checks.shape[0]):
    #     task_vector_recon[idx] = tv_scales[idx] * mask_m[idx] * mask_p[idx] * tensor_ones

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='./datasets').to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        task_vector_recon = torch.ones_like(tv_flat_checks[0])
        # scaled_mask_m, per_parameter_nums, total_parameter_nums = get_selected_parameter(mask_m[idx], pretrain_state, selected_list, thre=0.4)
        scaled_mask_m_p = get_half_parameter(mask_m[idx] * mask_p[idx], pretrain_state, selected_list, thre=0.4)
        # model_vector_recon = flat_ptm + scaled_mask_m * mask_p[idx] * tensor_ones * tv_scales[idx]
        model_vector_recon = flat_ptm + scaled_mask_m_p * tensor_ones * tv_scales[idx]


        merged_state_dict = vector_to_state_dict(model_vector_recon, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics

def get_t_switch_svd_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    # ****************** top alpha ************************
    alpha = 0.35

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)

    remove_keys = ['classifier.dense.weight', 'classifier.dense.bias' , 'classifier.out_proj.weight' , 'classifier.out_proj.bias']
    print(f"Flattening out Checkpoints")
    pretrain_state = merged_model.state_dict()
    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(), remove_keys) for check in models_to_merge])
    flat_ptm, filtered_keys = state_dict_to_vector(merged_model.state_dict(), remove_keys, sort_keys=True)
    tv_flat_checks = flat_ft - flat_ptm
    mask_p = torch.where(tv_flat_checks > 0, torch.tensor(1.0, device=tv_flat_checks.device), torch.tensor(-1.0, device=tv_flat_checks.device))
    tv_scales = torch.zeros_like(tv_flat_checks)
    mask_m = torch.zeros_like(tv_flat_checks, dtype=torch.bool)
    for idx in range(tv_flat_checks.size(0)):        
        pos_mask = tv_flat_checks[idx] > 0
        pos_values = tv_flat_checks[idx][pos_mask]
        if pos_values.numel() > 0:
            k_pos = int(pos_values.numel() * alpha)
            if k_pos > 0:
                topk_pos_values, topk_pos_indices = torch.topk(pos_values, k=k_pos)
                pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)
                selected_pos_indices = pos_indices[topk_pos_indices]
                mask_m[idx][selected_pos_indices] = True


        neg_mask = tv_flat_checks[idx]  < 0
        neg_values = tv_flat_checks[idx][neg_mask]
        if neg_values.numel() > 0:
            k_neg = int(neg_values.numel() * alpha)
            if k_neg > 0:

                bottomk_neg_values, bottomk_neg_indices = torch.topk(neg_values, k=k_neg, largest=False)
                neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(1)
                selected_neg_indices = neg_indices[bottomk_neg_indices]
                mask_m[idx][selected_neg_indices] = True

    tensor_ones = torch.ones_like(tv_flat_checks[0])

    tv_scales = torch.zeros(tv_flat_checks.shape[0])
    for idx in range(tv_flat_checks.shape[0]):
        tv_scales[idx] = torch.norm((flat_ft[idx] - flat_ptm) *  mask_m[idx], p=2) / torch.norm(mask_m[idx]* mask_p[idx], p=2)

    task_vector_recon = torch.ones_like(tv_flat_checks)

    for idx in range(tv_flat_checks.shape[0]):
        total_parameter_nums = 0
        total_parameter_nums_svd = 0
        offset_total = 0
        m_svd = torch.zeros(tv_flat_checks.shape[1])
        p_svd = torch.zeros(tv_flat_checks.shape[1])
        for n in pretrain_state:
            if n in filtered_keys:
                para_num = pretrain_state[n].numel()
                m_svd[offset_total: offset_total + para_num],  total_parameter_num_svd_m = our_mask_svd(mask_m[idx][offset_total: offset_total + para_num].reshape(pretrain_state[n].shape), density = 0.1)
                p_svd[offset_total: offset_total + para_num],  _ = our_mask_svd(mask_p[idx][offset_total: offset_total + para_num].reshape(pretrain_state[n].shape), density = 0.1)

                total_parameter_nums += para_num * 2
                total_parameter_nums_svd += total_parameter_num_svd_m *2 
                offset_total += para_num

        p_svd = torch.where(p_svd > 0, torch.tensor(1.0, device=tv_flat_checks.device), torch.tensor(-1.0, device=tv_flat_checks.device))
        task_vector_recon[idx] = tv_scales[idx] * m_svd * p_svd * tensor_ones

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )

        model_vector_recon = flat_ptm + task_vector_recon[idx]

        merged_state_dict = vector_to_state_dict(model_vector_recon, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics

def get_t_switch_layer_wise_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    # ****************** top alpha ************************
    alpha = 0.35

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)

    remove_keys = ['classifier.dense.weight', 'classifier.dense.bias' , 'classifier.out_proj.weight' , 'classifier.out_proj.bias']
    print(f"Flattening out Checkpoints")

    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(), remove_keys) for check in models_to_merge])
    flat_ptm, sorted_list = state_dict_to_vector(merged_model.state_dict(), remove_keys, sort_keys=True)
    filtered_keys = [k for k in sorted_list if "classifier" not in k]

    tv_flat_checks = flat_ft - flat_ptm
    mask_m = torch.zeros_like(tv_flat_checks, dtype=torch.bool)


    models_to_merge_state = [model.state_dict() for model in models_to_merge]
    pretrain_state = merged_model.state_dict()

    for idx in range(tv_flat_checks.size(0)):        
        pos_mask = tv_flat_checks[idx] > 0
        pos_values = tv_flat_checks[idx][pos_mask]
        if pos_values.numel() > 0:
            k_pos = int(pos_values.numel() * alpha)
            if k_pos > 0:

                topk_pos_values, topk_pos_indices = torch.topk(pos_values, k=k_pos)
                pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)
                selected_pos_indices = pos_indices[topk_pos_indices]
                mask_m[idx][selected_pos_indices] = True

 
        neg_mask = tv_flat_checks[idx]  < 0
        neg_values = tv_flat_checks[idx][neg_mask]
        if neg_values.numel() > 0:
            k_neg = int(neg_values.numel() * alpha)
            if k_neg > 0:

                bottomk_neg_values, bottomk_neg_indices = torch.topk(neg_values, k=k_neg, largest=False)
                neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(1)
                selected_neg_indices = neg_indices[bottomk_neg_indices]
                mask_m[idx][selected_neg_indices] = True

    mask_p = torch.where(tv_flat_checks > 0, torch.tensor(1.0, device=tv_flat_checks.device), torch.tensor(-1.0, device=tv_flat_checks.device))

    tv_scales = torch.zeros(tv_flat_checks.shape[0], len(filtered_keys))
    for idx in range(tv_flat_checks.shape[0]):
        offset_total = 0
        for idx_key, key in enumerate(filtered_keys):
            para_num_this_layer = models_to_merge_state[idx][key].numel()
            # tv_scales[idx][idx_key] = torch.norm(tv_flat_checks[idx][offset_total: offset_total+para_num_this_layer] *  mask_m[idx][offset_total: offset_total+para_num_this_layer], p=2) / torch.norm(mask_p[idx][offset_total: offset_total+para_num_this_layer] * mask_m[idx][offset_total: offset_total+para_num_this_layer], p=2)
            norm_mask_p_m = torch.norm(tv_flat_checks[idx][offset_total: offset_total + para_num_this_layer] * mask_m[idx][offset_total: offset_total + para_num_this_layer], p=2)
            norm_mask_p_m_masked = torch.norm(mask_p[idx][offset_total: offset_total + para_num_this_layer] * mask_m[idx][offset_total: offset_total + para_num_this_layer], p=2)
        

            if norm_mask_p_m_masked == 0:
                tv_scales[idx][idx_key] = 0
            else:
                tv_scales[idx][idx_key] = norm_mask_p_m / norm_mask_p_m_masked
            offset_total += para_num_this_layer

    tensor_ones = torch.ones_like(tv_flat_checks[0])

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        task_vector_recon = torch.ones_like(tv_flat_checks[0])
        
        offset_total = 0
        for idx_key, key in enumerate(filtered_keys):
            para_num_this_layer = pretrain_state[key].numel()
            task_vector_recon[offset_total: offset_total+para_num_this_layer] = flat_ptm[offset_total: offset_total+para_num_this_layer] + tv_scales[idx][idx_key] * mask_m[idx][offset_total: offset_total+para_num_this_layer] * mask_p[idx][offset_total: offset_total+para_num_this_layer] * tensor_ones[offset_total: offset_total+para_num_this_layer]
            # task_vector_recon[offset_total: offset_total+para_num_this_layer] = flat_ptm[offset_total: offset_total+para_num_this_layer] + tv_scales[idx] * mask_m[idx][offset_total: offset_total+para_num_this_layer] * mask_p[idx][offset_total: offset_total+para_num_this_layer] * tensor_ones[offset_total: offset_total+para_num_this_layer]
            offset_total += para_num_this_layer

        merged_state_dict = vector_to_state_dict(task_vector_recon, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics

def get_individual_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer):
    logger.info(f"configuration is {args}")

    set_random_seed(seed=0)

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        
        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"individual test performance on dataset {dataset_name}: {test_metrics}")

def get_personalized_mlp_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer):
    logger.info(f"configuration is {args}")
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')
    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # exclude parameter whose name matches "classifier"
    merged_params = merging_method.task_arithmetic(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"],  scaling_coefficient=0.3, 
                                                   ) 

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_params:
                if (".intermediate.dense." in param_name or ".output.dense." in param_name) and "attention" not in param_name:
                    param_value.data.copy_(model_to_merge.state_dict()[param_name])
                    print(param_name)
                else:
                    param_value.data.copy_(merged_params[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

def get_merge_DARE_performance(args: argparse.Namespace, models_to_merge: list,models_to_merge_ID: list,models_to_merge_OOD: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer, top_k_ratio=0.1):
    logger.info(f"configuration is {args}")

    # ****************** top alpha ************************
    alpha = 0.6

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

    set_random_seed(seed=0)

    scaling_coef_ = 1.0

    remove_keys = ['classifier.dense.weight', 'classifier.dense.bias' , 'classifier.out_proj.weight' , 'classifier.out_proj.bias']
    print(f"Flattening out Checkpoints")

    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(), remove_keys) for check in models_to_merge_ID])
    flat_ptm = state_dict_to_vector(merged_model.state_dict(), remove_keys)
    tv_flat_checks = flat_ft - flat_ptm

    for i in range(tv_flat_checks.shape[0]):  
        row = tv_flat_checks[i]  
        num_elements = row.numel()
        

        num_zero_elements = int(num_elements * alpha)
        

        zero_indices = torch.randperm(num_elements)[:num_zero_elements]
        

        tv_flat_checks[i][zero_indices] = 0

    task_vector_sum = sum(tv_flat_checks)
    merged_task_vector = scaling_coef_ * task_vector_sum
    flat_merged_mdoel = flat_ptm + merged_task_vector

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        try:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to('cpu')
        except:
            merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to('cpu')

        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )

        merged_state_dict = vector_to_state_dict(flat_merged_mdoel, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])

        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

    return test_metrics

if __name__ == "__main__":
    args.dataset_names = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
    # args.dataset_names = ["cola", "mrpc", "stsb", "mnli", "qnli", "sst2", "qqp", "rte"]
    # args.dataset_names = ["cola",  "rte"]
    args.dataset_names_ID = ["cola", "mrpc", "stsb", "mnli", "qnli"]
    args.dataset_names_OOD = ["sst2", "qqp", "rte"]
    assert all([dataset_name in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"] for dataset_name in args.dataset_names]), \
        'name in dataset_names must be contained in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]!'
    load_model_paths = []

    for dataset_name in args.dataset_names:
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        load_model_paths.append(f"./ckpts/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name))
    except:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir)
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)

    # load the checkpoint of each individual model that needs to be merged
    models_to_merge, trainers, = [], []
    models_to_merge_ID, trainers_ID, = [], []
    models_to_merge_OOD, trainers_OOD, = [], []
    for dataset_name, load_model_path in zip(args.dataset_names, load_model_paths):
        train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=dataset_name,
                                                                                             train_split_ratio_for_val=0.1,
                                                                                             max_seq_length=128)
        training_args = TrainingArguments(
            output_dir=load_model_path,                        # load model directory
            per_device_train_batch_size=args.batch_size,       # batch size per device during training
            per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
        )

        assert os.path.exists(os.path.join(training_args.output_dir, "trainer_state.json")), "cannot find file trainer_state.json!"
        model_to_merge = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=training_args.output_dir,
            num_labels=num_labels).to(args.device)
        trainer = CustomizedTrainer(
            model=model_to_merge,               # model to be merged
            args=training_args,                 # training arguments
            train_dataset=train_dataset,        # training dataset
            eval_dataset=test_dataset,          # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),   # function for computing metrics
            tokenizer=tokenizer                 # tokenizer
        )
        models_to_merge.append(model_to_merge.to('cpu'))
        trainers.append(trainer)
        if dataset_name in args.dataset_names_ID:
            models_to_merge_ID.append(model_to_merge.to('cpu'))
            trainers_ID.append(trainer)
        else:
            models_to_merge_OOD.append(model_to_merge.to('cpu'))
            trainers_OOD.append(trainer)


    for dataset_name, load_model_path in zip(args.dataset_names, load_model_paths):
        if dataset_name in args.dataset_names_ID:
            train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=dataset_name,
                                                                                                train_split_ratio_for_val=0.1,
                                                                                                max_seq_length=128)
            training_args = TrainingArguments(
                output_dir=load_model_path,                        # load model directory
                per_device_train_batch_size=args.batch_size,       # batch size per device during training
                per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
            )

            assert os.path.exists(os.path.join(training_args.output_dir, "trainer_state.json")), "cannot find file trainer_state.json!"
            model_to_merge = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=training_args.output_dir,
                num_labels=num_labels).to(args.device)
            trainer = CustomizedTrainer(
                model=model_to_merge,               # model to be merged
                args=training_args,                 # training arguments
                train_dataset=train_dataset,        # training dataset
                eval_dataset=test_dataset,          # evaluation dataset
                compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),   # function for computing metrics
                tokenizer=tokenizer                 # tokenizer
            )




    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    save_merge_log_path = f"./save_merge_logs/{args.merging_method_name}/{args.language_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)

    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")
    if args.merging_method_name == 'individual':
        performance = get_individual_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
    elif args.merging_method_name == 'our_svd_mask_merging':
        performance = get_our_svd_mask_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer, top_k_ratio = args.top_k_ratio)
    elif args.merging_method_name == 'our_svd_merging':
        performance = get_our_svd_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer, top_k_ratio = args.top_k_ratio)
    elif args.merging_method_name == 'ties_emr_merging':
        performance = get_ties_emr_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer, top_k_ratio = args.top_k_ratio)
    elif args.merging_method_name == 't_switch_merging':
        performance = get_t_switch_multi_msak_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer, top_k_ratio = args.top_k_ratio)
    elif args.merging_method_name == 't_switch_diff_merging':
        performance = get_t_switch_diff_vector_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer, top_k_ratio = args.top_k_ratio)
    elif args.merging_method_name == 't_switch_svd_merging':
        performance = get_t_switch_svd_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer, top_k_ratio = args.top_k_ratio)
    elif args.merging_method_name == 't_switch_module_merging':
        performance = get_our_t_switch_module_merge_performance_beifen(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer, top_k_ratio = args.top_k_ratio)
    elif args.merging_method_name == 'personalized_mlp':
        performance = get_personalized_mlp_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
    elif args.merging_method_name == 'our_svd_mask_merging_OOD':
        performance = get_our_svd_mask_merge_OOD_performance(args=args, models_to_merge=models_to_merge, models_to_merge_ID = models_to_merge_ID, models_to_merge_OOD = models_to_merge_OOD, trainers=trainers, trainers_OOD=trainers_OOD,  trainers_ID=trainers_ID, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
    elif args.merging_method_name == 'traditional_OOD':
        performance = get_merge_OOD_performance(args=args, models_to_merge=models_to_merge, models_to_merge_ID = models_to_merge_ID, models_to_merge_OOD = models_to_merge_OOD, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
    elif args.merging_method_name == 'DARE':
        performance = get_merge_DARE_performance(args=args, models_to_merge=models_to_merge, models_to_merge_ID = models_to_merge_ID, models_to_merge_OOD = models_to_merge_OOD, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
    else:
        performance = get_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
