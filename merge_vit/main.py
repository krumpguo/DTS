import os

import time
import sys
sys.path.append('merge_vit/')
import argparse
from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments
from ties_merging_utils import *

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def apply_vector(vector, pretrained_checkpoint):#, scaling_coef=1.0):
    """Apply a task vector to a pretrained model."""
    with torch.no_grad():
        pretrained_model = torch.load(pretrained_checkpoint, weights_only=False)
        new_state_dict = {}
        pretrained_state_dict = pretrained_model.state_dict()
        for key in pretrained_state_dict:
            if key not in vector:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                continue
            new_state_dict[key] = pretrained_state_dict[key] + vector[key]
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    return pretrained_model

## Model conversion utils
def state_dict_to_vector(state_dict, remove_keys=[], sort_keys=False):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    # sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
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
    # sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))
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

def our_svd_mask(
    tensor: torch.Tensor, 
    density: float,
    **kwargs,
):
    if density > 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)
    if tensor.ndimension() == 0:
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
    
    if tensor.ndimension() > 2:
        tensor_flat = tensor.reshape(-1) 
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

        tensor_mask_m = tensor_mask_m_flat.reshape(tensor.shape)
        tensor_mask_p = tensor_mask_p_flat.reshape(tensor.shape)
        tensor_tv_scales = tensor_tv_scales_flat.reshape(tensor.shape)
        tensor_ones = torch.ones_like(tensor)
        return tensor_mask_m * tensor_mask_p * tensor_tv_scales * tensor_ones, tensor.numel() * 3

    else:
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



args = parse_arguments()

exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD

args.model = 'ViT-B-32'
args.top_k_ratio = 1.0
args.device_num = 1

alpha = 0.3
print('*'*30 + 'svd coefficient:' + str(alpha) + '*'*30)
args.home = 'merge_vit'
model = args.model
args.data_location = 'merge_vit/data'
args.save = 'merge_vit/checkpoints/' + model
args.logs_path = 'merge_vit/logs/' + model
pretrained_checkpoint = 'merge_vit/checkpoints/'+model+'/zeroshot.pt'
args.device  = f'cuda:{args.device_num}'
top_k_ratio = args.top_k_ratio
print('*'*30 + 'personalized_top_k_ratio:' + str(top_k_ratio) + '*'*30)
str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_ties_merging.txt'.format(str_time_))
args.batch_size = 16

models_to_merge_state = [torch.load('merge_vit/checkpoints/'+model+'/'+dataset_name+'/finetuned.pt', weights_only=False).state_dict() for dataset_name in exam_datasets]
pretrain_model_state = torch.load(pretrained_checkpoint, weights_only=False).state_dict()

check_parameterNamesMatch(models_to_merge_state + [pretrain_model_state])

remove_keys = []
print(f"Flattening out Checkpoints")
flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in models_to_merge_state])
flat_ptm, sorted_list = state_dict_to_vector(pretrain_model_state, remove_keys, sort_keys=True)

tv_flat_checks = flat_ft - flat_ptm
assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, pretrain_model_state, remove_keys), pretrain_model_state)
assert all([check_state_dicts_equal(vector_to_state_dict(flat_ft[i], pretrain_model_state, remove_keys), models_to_merge_state[i])for i in range(len(models_to_merge_state))])


K = 20
merge_func = "dis-sum"
scaling_coef_ = 0.3

# merged_tv = torch.sum(tv_flat_checks, dim=0) 
merged_tv = ties_merging(tv_flat_checks, reset_thresh=K, merge_func=merge_func,)
flat_merged = merged_tv * scaling_coef_ + flat_ptm
merged_state = vector_to_state_dict(flat_merged, pretrain_model_state, remove_keys=remove_keys)

mean_parameter_memory = 0
mean_parameter_memory_svd = 0

print(f"reconstructing and evaluating merged model")

accs = []
for idx_dataname, dataset in enumerate(exam_datasets):
    image_encoder = torch.load(pretrained_checkpoint, weights_only=False)
    total_parameter_nums = 0
    total_parameter_nums_svd = 0
    for idx, (param_name, param_value) in enumerate(image_encoder.named_parameters()):
        if param_name in pretrain_model_state:

            svd_param, total_parameter_num_svd = our_svd_mask(models_to_merge_state[idx_dataname][param_name] - pretrain_model_state[param_name], 0.3)
            total_parameter_nums += pretrain_model_state[param_name].numel()
            total_parameter_nums_svd += total_parameter_num_svd
            param_value.data.copy_(svd_param + pretrain_model_state[param_name])
             
    mean_parameter_memory += total_parameter_nums/1024/1024 * 4
    mean_parameter_memory_svd += total_parameter_nums_svd/1024/1024/8

    metrics = eval_single_dataset(image_encoder, dataset, args)
    print(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
    accs.append(metrics.get('top1')*100)

times_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


