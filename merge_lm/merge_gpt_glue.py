import datasets.arrow_dataset
from tqdm import tqdm
import numpy
from datasets import load_dataset, load_from_disk
import copy
import os
import sys
from collections import OrderedDict
import transformers
from utils.utils import set_random_seed
from model_merging_methods.merging_methods import MergingMethod

import sys
import json
import argparse
from torch.utils.data import DataLoader
import time
import logging
from functools import partial
from torchmetrics import Accuracy
import torch
from transformers import  GPT2Tokenizer #, GPT2ForSequenceClassification
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.glue_data_loader import GLUEDataLoader
from utils.metrics import compute_metrics
from utils.customized_trainers import CustomizedTrainer
from model_merging_methods.mask_weights_utils import mask_model_weights
from utils.load_config import cache_dir

from transformers import (
    GPT2ForSequenceClassification,
    GPT2Model,
    GPT2Tokenizer,
    default_data_collator,
    AutoConfig
)

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
    sorted_reference_dict = OrderedDict((reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict

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
    
    # U, S, V = torch.svd(tensor)
    # S = (S >= S[int(len(S) * density)]) * S
    # res = U @ torch.diag(S) @ V.T

    # `torch.linalg.svd()`: good for dense matrix
    # `torch.svd()`: deprecated
    # `torch.svd_lowrank()`: good for huge sparse matrix
    driver = None
    if tensor.is_cuda:
        driver = 'gesvda'
    
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=True, driver=driver)
    new_rank = int(density * len(S))
    U, S, Vh = U[:, :new_rank], S[:new_rank], Vh[:new_rank, :]
    U
    res = U @ torch.diag(S) @ Vh

    return res, U.numel() + S.numel() + Vh.numel()

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




def get_personalized_mlp_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list,
                          tokenizer: transformers.AutoTokenizer, logger, top_k_ratio= 0.1):
    logger.info(f"configuration is {args}")
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # exclude parameter whose name matches "classifier"
    merged_params = merging_method.task_arithmetic(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*score.*"],
                                                   scaling_coefficient=0.3)
    print(f"Flattening out Checkpoints")
    models_to_merge_state_dict = [check.state_dict() for check in models_to_merge]

    averge_acc = []
    for idx, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models_to_merge)):
        merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')
        merged_model.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_name}") # load the config

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_params:
                if '.mlp.' in param_name:
                    param_value.data.copy_(model_to_merge.state_dict()[param_name])
                    print(param_name)
                else:
                    param_value.data.copy_(merged_params[param_name])

        merged_model.score = model_to_merge.score
        merged_model.to(args.device)
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)

        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        with torch.no_grad():
            accuracy = Accuracy("multiclass", num_classes=num_labels[
                dataset_name])  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
            loader = DataLoader(
                ds_val,
                collate_fn=default_data_collator,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
            )
            for batch in (
                    pbar := tqdm(
                        loader, desc="Evaluating", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)

                outputs = merged_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                pbar.set_postfix({"accuracy": acc.item()})

            acc = accuracy.compute().item()
            logger.info(f"acc on {dataset_name}: {acc}")
            averge_acc.append(acc)

    logger.info(f"average acc on all datasets: {sum(averge_acc) / len(averge_acc)}")

def get_our_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list,
                          tokenizer: transformers.AutoTokenizer, logger, top_k_ratio= 0.1):
    logger.info(f"configuration is {args}")
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to(args.device)

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # exclude parameter whose name matches "classifier"
    merged_params = merging_method.merging_models(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*score.*"],
                                                   models_use_deepcopy=True)
    remove_keys = ['score.weight']
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
                    norm_single_neuron = torch.cat((merged_tv[offset_total: offset_total + cols], merged_tv[bias_location].unsqueeze(0)))
                    # diffs= torch.norm(single_neuron) / torch.norm(norm_single_neuron)
                    diffs= torch.norm(single_neuron) / cols
                    # diffs = torch.mean(torch.abs(single_neuron)) 
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
                    norm_single_neuron = merged_tv[offset_total: offset_total + cols]
                    # diffs= torch.norm(single_neuron) / torch.norm(norm_single_neuron)
                    diffs= torch.norm(single_neuron) / cols
                    # diffs = torch.mean(torch.abs(single_neuron)) 
                    avg_abs_diffs[idx_task, offset_row] = diffs
                offset_row += 1
                offset_total += cols
            layer_num_location += 1

    mask = torch.zeros_like(avg_abs_diffs, dtype=torch.bool)

    selected_neuron = int(total_neuron_num * top_k_ratio)


    Total_memory_bytes = 0
    for i in range(n_models):
        if selected_neuron > 0:
            _, idxs = torch.topk(avg_abs_diffs[i], selected_neuron)
            mask[i, idxs] = True
        total_params = 0       # 
        total_bytes  = 0       # 
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
                        param = flat_ptm[offset_total: offset_total + cols]   # 
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
                        param = flat_ptm[offset_total: offset_total + cols]    # 
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

    averge_acc = []
    for idx, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models_to_merge)):
        
        merged_model.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_name}") # load the config

        task_vector_recon = flat_merged * (~mask_merge[idx]) + flat_ft[idx] * mask_merge[idx]
        merged_state_dict = vector_to_state_dict(task_vector_recon, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])


        merged_model.score = model_to_merge.score
        merged_model.to(args.device)
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)

        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        with torch.no_grad():
            accuracy = Accuracy("multiclass", num_classes=num_labels[
                dataset_name])  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
            loader = DataLoader(
                ds_val,
                collate_fn=default_data_collator,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
            )
            for batch in (
                    pbar := tqdm(
                        loader, desc="Evaluating", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)

                outputs = merged_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                pbar.set_postfix({"accuracy": acc.item()})

            acc = accuracy.compute().item()
            logger.info(f"acc on {dataset_name}: {acc}")
            averge_acc.append(acc)

    logger.info(f"average acc on all datasets: {sum(averge_acc) / len(averge_acc)}")


def get_our_svd_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list,
                          tokenizer: transformers.AutoTokenizer, logger, top_k_ratio= 0.1):
    logger.info(f"configuration is {args}")
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # exclude parameter whose name matches "classifier"
    merged_params = merging_method.task_arithmetic(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*score.*"],
                                                   scaling_coefficient=0.3)

    models_to_merge_state = [model.state_dict() for model in models_to_merge]

    mean_parameter_memory = 0
    mean_parameter_memory_svd = 0
    averge_acc = []
    for idx_dataname, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models_to_merge)):
        merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')
        merged_model.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_name}") # load the config

        total_parameter_nums = 0
        total_parameter_nums_svd = 0
        for idx, (param_name, param_value) in enumerate(merged_model.named_parameters()):
            if param_name in merged_params:
                svd_param, total_parameter_num_svd = our_svd(models_to_merge_state[idx_dataname][param_name] - merged_params[param_name], 0.03)
                total_parameter_nums += merged_params[param_name].numel()
                total_parameter_nums_svd += total_parameter_num_svd
                param_value.data.copy_(svd_param + merged_params[param_name])

        mean_parameter_memory += total_parameter_nums/1024/1024 * 4
        mean_parameter_memory_svd += total_parameter_nums_svd/1024/1024 * 4

        merged_model.score = model_to_merge.score
        merged_model.to(args.device)
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)

        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        with torch.no_grad():
            accuracy = Accuracy("multiclass", num_classes=num_labels[
                dataset_name])  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
            loader = DataLoader(
                ds_val,
                collate_fn=default_data_collator,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
            )
            for batch in (
                    pbar := tqdm(
                        loader, desc="Evaluating", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)

                outputs = merged_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                pbar.set_postfix({"accuracy": acc.item()})

            acc = accuracy.compute().item()
            logger.info(f"acc on {dataset_name}: {acc}")
            averge_acc.append(acc)

    logger.info(f"average acc on all datasets: {sum(averge_acc) / len(averge_acc)}")


def get_our_svd_mask_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list,
                          tokenizer: transformers.AutoTokenizer, logger, top_k_ratio= 0.1):
    logger.info(f"configuration is {args}")
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')

    set_random_seed(seed=0)

    merged_params = merging_method.ties_merging(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*score.*"],
                                                   scaling_coefficient=0.3)
    models_to_merge_state = [model.state_dict() for model in models_to_merge]
    pretrain_model_state = merged_model.state_dict()

    mean_parameter_memory = 0
    mean_parameter_memory_svd = 0
    averge_acc = []
    for idx_dataname, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models_to_merge)):
        merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')
        merged_model.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_name}") # load the config

        total_parameter_nums = 0
        total_parameter_nums_svd = 0
        for idx, (param_name, param_value) in enumerate(merged_model.named_parameters()):
            if param_name in merged_params:

                svd_param, total_parameter_num_svd = our_svd_mask_2(models_to_merge_state[idx_dataname][param_name] - pretrain_model_state[param_name], 0.1)
                total_parameter_nums += pretrain_model_state[param_name].numel()
                total_parameter_nums_svd += total_parameter_num_svd
                param_value.data.copy_(svd_param + pretrain_model_state[param_name])


        mean_parameter_memory += total_parameter_nums/1024/1024 * 4
        mean_parameter_memory_svd += total_parameter_nums_svd/1024/1024/8

        merged_model.score = model_to_merge.score
        merged_model.to(args.device)
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)

        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        with torch.no_grad():
            accuracy = Accuracy("multiclass", num_classes=num_labels[
                dataset_name])  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
            loader = DataLoader(
                ds_val,
                collate_fn=default_data_collator,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
            )
            for batch in (
                    pbar := tqdm(
                        loader, desc="Evaluating", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)

                outputs = merged_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                pbar.set_postfix({"accuracy": acc.item()})

            acc = accuracy.compute().item()
            logger.info(f"acc on {dataset_name}: {acc}")
            averge_acc.append(acc)

    logger.info(f"average acc on all datasets: {sum(averge_acc) / len(averge_acc)}")

def get_t_switch_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list,
                          tokenizer: transformers.AutoTokenizer, logger, top_k_ratio= 0.1):
    logger.info(f"configuration is {args}")
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')
     
    alpha = 0.3

    remove_keys = ['score.weight']
    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # exclude parameter whose name matches "classifier"
    
    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(),  remove_keys) for check in models_to_merge])
    flat_ptm = state_dict_to_vector(merged_model.state_dict(), remove_keys)

    tv_flat_checks = flat_ft - flat_ptm

    mask_p = torch.where(tv_flat_checks > 0, torch.tensor(1.0, device=tv_flat_checks.device), torch.tensor(-1.0, device=tv_flat_checks.device))
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
        tv_scales[idx] = torch.norm(tv_flat_checks[idx] *  mask_m[idx], p=2) / torch.norm(mask_m[idx]* mask_p[idx], p=2)

    task_vector_recon = torch.ones_like(tv_flat_checks)
    for idx in range(tv_flat_checks.shape[0]):
        task_vector_recon[idx] = tv_scales[idx] * mask_m[idx] * mask_p[idx] * tensor_ones

    averge_acc = []
    for idx_dataname, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models_to_merge)):
        merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')
        merged_model.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_name}") # load the config

        model_vector_recon = flat_ptm + task_vector_recon[idx_dataname]
        merged_state_dict = vector_to_state_dict(model_vector_recon, merged_model.state_dict(), remove_keys)
        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])
        merged_model.score = model_to_merge.score
        merged_model.to(args.device)
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)

        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        with torch.no_grad():
            accuracy = Accuracy("multiclass", num_classes=num_labels[
                dataset_name])  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
            loader = DataLoader(
                ds_val,
                collate_fn=default_data_collator,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
            )
            for batch in (
                    pbar := tqdm(
                        loader, desc="Evaluating", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)

                outputs = merged_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                pbar.set_postfix({"accuracy": acc.item()})

            acc = accuracy.compute().item()
            logger.info(f"acc on {dataset_name}: {acc}")
            averge_acc.append(acc)

    logger.info(f"average acc on all datasets: {sum(averge_acc) / len(averge_acc)}")



def get_merge_DARE_performance(args: argparse.Namespace, models_to_merge: list, trainers: list,
                          tokenizer: transformers.AutoTokenizer, logger, top_k_ratio= 0.1):
    logger.info(f"configuration is {args}")
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')
    alpha = 0.6
    scaling_coef_ = 0.6
    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)

    averge_acc = []
    
    remove_keys = ['score.weight']
    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack([state_dict_to_vector(check.state_dict(), remove_keys) for check in models_to_merge])
    flat_ptm, sorted_list = state_dict_to_vector(merged_model.state_dict(), remove_keys, sort_keys=True)

    tv_flat_checks = flat_ft - flat_ptm

    for i in range(tv_flat_checks.shape[0]):  # 
        row = tv_flat_checks[i]  # 
        num_elements = row.numel()

        num_zero_elements = int(num_elements * alpha)
        

        zero_indices = torch.randperm(num_elements)[:num_zero_elements]

        tv_flat_checks[i][zero_indices] = 0

    task_vector_sum = sum(tv_flat_checks)
    merged_task_vector = scaling_coef_ * task_vector_sum
    flat_merged_mdoel = flat_ptm + merged_task_vector
    merged_state_dict = vector_to_state_dict(flat_merged_mdoel, merged_model.state_dict(), remove_keys=remove_keys)

    for idx, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models_to_merge)):
        
        merged_model.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_name}") # load the config
    
        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])

        merged_model.score = model_to_merge.score
        merged_model.to(args.device)
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)

        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        with torch.no_grad():
            accuracy = Accuracy("multiclass", num_classes=num_labels[
                dataset_name])  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
            loader = DataLoader(
                ds_val,
                collate_fn=default_data_collator,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
            )
            for batch in (
                    pbar := tqdm(
                        loader, desc="Evaluating", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)

                outputs = merged_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                pbar.set_postfix({"accuracy": acc.item()})

            acc = accuracy.compute().item()
            logger.info(f"acc on {dataset_name}: {acc}")
            averge_acc.append(acc)

    logger.info(f"average acc on all datasets: {sum(averge_acc) / len(averge_acc)}")



def get_our_merge_performance_0(args: argparse.Namespace, models_to_merge: list, trainers: list,
                          tokenizer: transformers.AutoTokenizer, logger, top_k_ratio= 0.1):
    logger.info(f"configuration is {args}")
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to(args.device)

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # exclude parameter whose name matches "classifier"
    merged_params = merging_method.merging_models(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*score.*"],
                                                   models_use_deepcopy=True)
    averge_acc = []
    
    remove_keys = ['score.weight']
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
        if merged_model_dict[n].dim()==0:
            total_neuron_num +=1
            neuron_num_per_layer.append(1)
            column_num_per_layer.append(1)
        elif merged_model_dict[n].dim()==1:
            total_neuron_num +=1
            neuron_num_per_layer.append(1)
            column_num_per_layer.append(merged_model_dict[n].size(0))
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

    selected_neuron = int(total_neuron_num * top_k_ratio)


    Total_memory_bytes = 0
    for i in range(n_models):
        if selected_neuron > 0:

            _, idxs = torch.topk(avg_abs_diffs[i], selected_neuron)
            mask[i, idxs] = True
        total_params = 0       # 
        total_bytes  = 0       # 
        offset_row = 0
        offset_total = 0
        for idx_key, k in enumerate(filtered_keys):
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
    for idx_key, k in enumerate(filtered_keys):  
        rows = neuron_num_per_layer[idx_key]
        cols = column_num_per_layer[idx_key]
        for idx_row in range(rows):
            for idx_task in range(n_models):
                m = mask[idx_task][offset_row]   
                mask_merge[idx_task][offset_total: offset_total + cols] = m
            offset_row += 1
            offset_total += cols


    for idx, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models_to_merge)):
        
        merged_model.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_name}") # load the config

        task_vector_recon = flat_merged * (~mask_merge[idx]) + flat_ft[idx] * mask_merge[idx]
        merged_state_dict = vector_to_state_dict(task_vector_recon, merged_model.state_dict(), remove_keys=remove_keys)

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_state_dict:
                param_value.data.copy_(merged_state_dict[param_name])


        merged_model.score = model_to_merge.score
        merged_model.to(args.device)
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)

        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        with torch.no_grad():
            accuracy = Accuracy("multiclass", num_classes=num_labels[
                dataset_name])  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
            loader = DataLoader(
                ds_val,
                collate_fn=default_data_collator,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
            )
            for batch in (
                    pbar := tqdm(
                        loader, desc="Evaluating", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)

                outputs = merged_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                pbar.set_postfix({"accuracy": acc.item()})

            acc = accuracy.compute().item()
            logger.info(f"acc on {dataset_name}: {acc}")
            averge_acc.append(acc)

    logger.info(f"average acc on all datasets: {sum(averge_acc) / len(averge_acc)}")

def get_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list,
                          tokenizer: transformers.AutoTokenizer, logger):
    logger.info(f"configuration is {args}")
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to(args.device)

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # exclude parameter whose name matches "classifier"
    merged_params = merging_method.merging_models(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*score.*"],
                                                   models_use_deepcopy=True)
    averge_acc = []

    for idx, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models_to_merge)):
        # merged_model.config =
        merged_model.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_name}") # load the config

        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_params:
                param_value.data.copy_(merged_params[param_name])
        merged_model.score = model_to_merge.score
        merged_model.to(args.device)
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)

        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        with torch.no_grad():
            accuracy = Accuracy("multiclass", num_classes=num_labels[
                dataset_name])  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
            loader = DataLoader(
                ds_val,
                collate_fn=default_data_collator,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
            )
            for batch in (
                    pbar := tqdm(
                        loader, desc="Evaluating", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)

                outputs = merged_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                pbar.set_postfix({"accuracy": acc.item()})

            acc = accuracy.compute().item()
            logger.info(f"acc on {dataset_name}: {acc}")
            averge_acc.append(acc)

    logger.info(f"average acc on all datasets: {sum(averge_acc) / len(averge_acc)}")

def get_pretrain_performance(args: argparse.Namespace, models_to_merge: list, trainers: list,
                          tokenizer: transformers.AutoTokenizer, logger):
    logger.info(f"configuration is {args}")
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to(args.device)

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # exclude parameter whose name matches "classifier"

    averge_acc = []

    for idx, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models_to_merge)):
        # merged_model.config =
        merged_model.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_name}") # load the config

        merged_model.score = model_to_merge.score
        merged_model.to(args.device)
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)

        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        with torch.no_grad():
            accuracy = Accuracy("multiclass", num_classes=num_labels[
                dataset_name])  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
            loader = DataLoader(
                ds_val,
                collate_fn=default_data_collator,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
            )
            for batch in (
                    pbar := tqdm(
                        loader, desc="Evaluating", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)

                outputs = merged_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                pbar.set_postfix({"accuracy": acc.item()})

            acc = accuracy.compute().item()
            logger.info(f"acc on {dataset_name}: {acc}")
            averge_acc.append(acc)

    logger.info(f"average acc on all datasets: {sum(averge_acc) / len(averge_acc)}")

def get_individual_performance(args: argparse.Namespace, models_to_merge: list, trainers: list,
                          tokenizer: transformers.AutoTokenizer, logger):
    logger.info(f"configuration is {args}")
    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # exclude parameter whose name matches "classifier"

    averge_acc=[]

    for idx, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models_to_merge)):
        # merged_model.config =
        model_to_merge.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_name}") # load the config

        model_to_merge.to(args.device)
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)

        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        with torch.no_grad():
            accuracy = Accuracy("multiclass", num_classes=num_labels[
                dataset_name])  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
            loader = DataLoader(
                ds_val,
                collate_fn=default_data_collator,
                batch_size=args.batch_size,
                num_workers=4,
                shuffle=True,
            )
            for batch in (
                    pbar := tqdm(
                        loader, desc="Evaluating", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)

                outputs = model_to_merge(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                pbar.set_postfix({"accuracy": acc.item()})

            acc = accuracy.compute().item()
            logger.info(f"acc on {dataset_name}: {acc}")
            averge_acc.append(acc)

    logger.info(f"average acc on all datasets: {sum(averge_acc) / len(averge_acc)}")


def mrpc_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples['sentence1'],#, 'sentence2'],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def mnli_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def cola_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def qnli_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question"],
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def qqp_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question1"],
        examples["question2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs

class TokenizedGLUE:
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def load_dataset(
        self, name
    ):
        glue_dataset_loaders = {
            "mrpc": self.load_mrpc_dataset,
            "mnli": self.load_mnli_dataset,
            "cola": self.load_cola_dataset,
            "sst2": self.load_sst2_dataset,
            "qnli": self.load_qnli_dataset,
            "qqp": self.load_qqp_dataset,
            "rte": self.load_rte_dataset,
            # "wnli": load_wnli_dataset,
        }
        return glue_dataset_loaders[name]()

    
    def load_mrpc_dataset(self):
        local_path = 'merge_lm/datasets/gpt2/glue/mrpc'
        try:

            dataset = load_from_disk(local_path)
            print("Loaded MRPC dataset from local disk.")
        except Exception as e:
            print(f"Failed to load local MRPC dataset: {e}")
            print("Falling back to online download via Hugging Face Hub...")
            dataset = load_dataset("glue", "mrpc", cache_dir=cache_dir)
            
            print(f"Saving MRPC dataset to {local_path} ...")
            dataset.save_to_disk(local_path)

        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=['sentence1', 'sentence2'],
        )
        return dataset


    def load_rte_dataset(self):
        local_path = 'merge_lm/datasets/gpt2/glue/rte'
        try:

            dataset = load_from_disk(local_path)
            print("Loaded RTE dataset from local disk.")
        except Exception as e:
            print(f"Failed to load local RTE dataset: {e}")
            print("Falling back to online download via Hugging Face Hub...")
            dataset = load_dataset("glue", "rte", cache_dir=cache_dir)
            
            print(f"Saving RTE dataset to {local_path} ...")
            dataset.save_to_disk(local_path)

        dataset = dataset.map(
            # RTE has the same format as MRPC
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset


    def load_wnli_dataset(self):
        local_path = 'merge_lm/datasets/gpt2/glue/wnli'
        try:

            dataset = load_from_disk(local_path)
            print("Loaded WNLI dataset from local disk.")
        except Exception as e:
            print(f"Failed to load local WNLI dataset: {e}")
            print("Falling back to online download via Hugging Face Hub...")
            dataset = load_dataset("glue", "wnli", cache_dir=cache_dir)
            
            print(f"Saving WNLI dataset to {local_path} ...")
            dataset.save_to_disk(local_path)

        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset


    def load_qqp_dataset(self):
        local_path = 'merge_lm/datasets/gpt2/glue/qqp'
        try:

            dataset = load_from_disk(local_path)
            print("Loaded QQP dataset from local disk.")
        except Exception as e:
            print(f"Failed to load local QQP dataset: {e}")
            print("Falling back to online download via Hugging Face Hub...")
            dataset = load_dataset("glue", "qqp", cache_dir=cache_dir)
            
            print(f"Saving QQP dataset to {local_path} ...")
            dataset.save_to_disk(local_path)
        dataset = dataset.map(
            partial(qqp_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=['question1', 'question2'],
        )
        return dataset


    def load_mnli_dataset(self):
        local_path = 'merge_lm/datasets/gpt2/glue/mnli'
        try:

            dataset = load_from_disk(local_path)
            print("Loaded MNLI dataset from local disk.")
        except Exception as e:
            print(f"Failed to load local MNLI dataset: {e}")
            print("Falling back to online download via Hugging Face Hub...")
            dataset = load_dataset("glue", "mnli",  cache_dir=cache_dir)
            
            print(f"Saving MNLI dataset to {local_path} ...")
            dataset.save_to_disk(local_path)
        dataset = dataset.map(
            partial(mnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["premise", "hypothesis"],
        )
        return dataset


    def load_cola_dataset(self):
        local_path = 'merge_lm/datasets/gpt2/glue/cola'
        try:

            dataset = load_from_disk(local_path)
            print("Loaded CoLA dataset from local disk.")
        except Exception as e:
            print(f"Failed to load local CoLA dataset: {e}")
            print("Falling back to online download via Hugging Face Hub...")
            dataset = load_dataset("glue", "cola", cache_dir=cache_dir)
            
            print(f"Saving CoLA dataset to {local_path} ...")
            dataset.save_to_disk(local_path)
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset

    def load_sst2_dataset(self):
        local_path = 'merge_lm/datasets/gpt2/glue/sst2'
        try:

            dataset = load_from_disk(local_path)
            print("Loaded SST2 dataset from local disk.")
        except Exception as e:
            print(f"Failed to load local SST2 dataset: {e}")
            print("Falling back to online download via Hugging Face Hub...")
            dataset = load_dataset("glue", "sst2", cache_dir=cache_dir)
            
            print(f"Saving SST2 dataset to {local_path} ...")
            dataset.save_to_disk(local_path)
        print(dataset.column_names)
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset


    def load_qnli_dataset(self):
        local_path = 'merge_lm/datasets/gpt2/glue/qnli'
        try:

            dataset = load_from_disk(local_path)
            print("Loaded QNLI dataset from local disk.")
        except Exception as e:
            print(f"Failed to load local QNLI dataset: {e}")
            print("Falling back to online download via Hugging Face Hub...")
            dataset = load_dataset("glue", "qnli", cache_dir=cache_dir)
            
            print(f"Saving QNLI dataset to {local_path} ...")
            dataset.save_to_disk(local_path)
        print(dataset.column_names)
        dataset = dataset.map(
            partial(qnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["question", "sentence"],
        )
        return dataset


num_labels = {
        'cola': 2,
        'sst2': 2,
        'mrpc': 2,
        'stsb': 5,
        'qqp': 2,
        'mnli': 3,
        'qnli': 2,
        'rte': 2
    }
dataset_names = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]
# dataset_names = ["mrpc", "sst2"] 
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Interface for inference PLMs on glue")
    parser.add_argument("--language_model_name", type=str, default="gpt2", help="name of the language model", choices=["gpt2"])
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--merging_method_name", type=str, default="our_svd_merging",
                        help="name of the method to merge models",
                        choices=["ties_merging", "task_arithmetic", "fisher_merging", "regmean_merging", "individual", "our_merging", "our_svd_merging", "our_svd_mask_merging"])
    parser.add_argument('--ckpt_path', type=str, default='merge_lm/ckpts/gpt2',help="ckpt path")
    parser.add_argument("--gpu", type=int, default=6, help="number of gpu to use")
    parser.add_argument("--top_k_ratio", type=float, default=1, help='Top K ratio for parameter selection')


    try:
        args = parser.parse_args()
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    except:
        parser.print_help()
        sys.exit()
    args.dataset_names = dataset_names

    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2')

    tokenizer.model_max_length = 512
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)
    pretrained_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')

    models = []
    loaders = []
    for dataset_name in dataset_names:
        args.dataset_name = dataset_name
        load_model_path = args.ckpt_path+f"/gpt2_{dataset_name}"
        finetuned_model = GPT2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=load_model_path).to('cpu')
        models.append(finetuned_model)
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
    logger.info(f"configuration is {args}")
    
    if args.merging_method_name == 'individual':
        performance = get_individual_performance(args, models, loaders, tokenizer, logger)
    elif args.merging_method_name == 'pretrain':
        performance = get_pretrain_performance(args, models, loaders, tokenizer, logger)
    elif args.merging_method_name == 'our_merging':
        performance = get_our_merge_performance(args, models, loaders, tokenizer, logger, args.top_k_ratio)
    elif args.merging_method_name == 'our_svd_merging':
        performance = get_our_svd_merge_performance(args, models, loaders, tokenizer, logger, args.top_k_ratio)    
    elif args.merging_method_name == 'our_svd_mask_merging':
        performance = get_our_svd_mask_merge_performance(args, models, loaders, tokenizer, logger, args.top_k_ratio)   
    elif args.merging_method_name == 't_switch_merging':
        performance = get_t_switch_merge_performance(args, models, loaders, tokenizer, logger, args.top_k_ratio) 
    elif args.merging_method_name == 'personalized_mlp':
        performance = get_personalized_mlp_merge_performance(args, models, loaders, tokenizer, logger, args.top_k_ratio) 
    elif args.merging_method_name == 'DARE':
        performance = get_merge_DARE_performance(args, models, loaders, tokenizer, logger, args.top_k_ratio) 
    else:
        performance = get_merge_performance(args, models, loaders, tokenizer, logger)
