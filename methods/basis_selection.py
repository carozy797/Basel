import math
import json
import os
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._tensor import Tensor
from torch.nn.modules.container import Sequential
from torch.nn.modules.linear import Linear
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel
import torch.distributed as dist


class BSLinear(nn.Module):
    def __init__(self, U, Vh, S, additional_dim, bias=None) -> None:
        super(BSLinear, self).__init__()
        output_dim = U.size(0)
        inner_dim = Vh.size(0)
        pad_dim = math.ceil(inner_dim / 16) * 16 - inner_dim
        inner_dim = inner_dim + pad_dim
        input_dim = Vh.size(1)
        self.output_dim = output_dim
        self.inner_dim = inner_dim
        self.input_dim = input_dim
        device = Vh.device
        self.device = device
        if pad_dim == 0:
            self.weight = nn.Parameter(S.clone() ** 0.5)
        else:
            self.weight = nn.Parameter(torch.cat([(S.clone() ** 0.5), torch.zeros(pad_dim, device=device)]))
        if pad_dim == 0:
            self.register_buffer("U", U.clone().contiguous().to(torch.bfloat16))
            self.register_buffer("Vh", Vh.clone().contiguous().to(torch.bfloat16))
        else:
            self.register_buffer("U", torch.cat([U, torch.zeros(output_dim, pad_dim, device=device)], dim=1).to(torch.bfloat16))
            self.register_buffer("Vh", torch.cat([Vh, torch.zeros(pad_dim, input_dim, device=device)], dim=0).to(torch.bfloat16))
        if bias is not None:
            self.bias = nn.Parameter(bias.data.clone())
        else:
            self.bias = None
        self.register_buffer("mask", torch.cat([torch.ones(inner_dim-pad_dim, device=device), torch.zeros(pad_dim, device=device)]))
        self.U_additional = nn.Parameter(
            torch.zeros(output_dim, additional_dim, device=device)
        )
        scale = 1.0 if additional_dim == 0 else 1 / math.sqrt(additional_dim)
        self.Vh_additional = nn.Parameter(
            torch.randn(additional_dim, input_dim, device=device) * scale
        )


    def forward(self, input: Tensor) -> Tensor:
        outputs = F.linear(
            input,
            torch.matmul(
                torch.matmul(
                    self.U,
                    torch.diag((self.weight ** 2) * self.mask)
                ),
                self.Vh
            )
            + torch.matmul(
                self.U_additional,
                self.Vh_additional
            )
        )       
        return outputs


def model_init(model, additional_dim):
    for name, layer in model._modules.items():
        if len(list(layer.children())) > 0:
            model._modules[name] = model_init(layer, additional_dim)
        else:
            if type(layer) == nn.Linear:
                model._modules[name] = layer_init(layer, additional_dim)
        torch.cuda.empty_cache()
    return model


def layer_init(layer, additional_dim) -> BSLinear:
    if layer.bias is not None:
        bias = layer.bias.data
    else:
        bias = None
    U, S, Vh = torch.linalg.svd(layer.weight.data)
    rank = rank_cal(S)
    new_layer = BSLinear(U[:, :rank], Vh[:rank, :], S[:rank], additional_dim, bias=bias)
    return new_layer


def rank_cal(S, eps: float = 1e-3) -> Tensor:
    rank = torch.sum(S > (torch.mean(S) * eps))
    return rank


def convert_from_base_to_factorized_model(
    model,
    dim_dict,
    prefix: str = ""
):
    for name, layer in model._modules.items():
        if len(list(layer.children())) > 0:
            model._modules[name] = convert_from_base_to_factorized_model(
                layer, dim_dict, prefix + name + "."
            )
        else:
            if type(layer) == nn.Linear:
                model._modules[name] = convert_base_to_factorized_layer(
                    layer, prefix + name, dim_dict
                )
        del layer
        torch.cuda.empty_cache()
    return model

def convert_base_to_factorized_layer(
    layer,
    name,
    dim_dict,
) -> Sequential:
    bias_flag = layer.bias is not None
    dim0 = dim_dict[name + ".0"]
    dim1 = dim_dict[name + ".1"]
    first_layer = torch.nn.Linear(
        in_features=dim0[1], out_features=dim0[0], bias=False
    )
    second_layer = torch.nn.Linear(
        in_features=dim1[1], out_features=dim1[0], bias=bias_flag
    )
    if bias_flag:
        second_layer.bias.data = layer.bias.data.clone()
    new_layers = [first_layer, second_layer]
    return nn.Sequential(*new_layers)


def convert_to_factorized_model(model):
    for name, layer in model._modules.items():
        if len(list(layer.children())) > 0:
            model._modules[name] = convert_to_factorized_model(layer)
        else:
            if type(layer) == BSLinear:
                model._modules[name] = convert_to_factorized_layer(layer)
        del layer
        torch.cuda.empty_cache()
    return model


def convert_to_factorized_layer(layer) -> Sequential:
    bias_flag = layer.bias is not None
    weight = (
        torch.matmul(
            torch.matmul(
                layer.U.to(layer.weight.dtype).cuda(),
                torch.diag((layer.weight.cuda()**2) * layer.mask.cuda()),
            ),
            layer.Vh.to(layer.weight.dtype).cuda(),
        )
        + torch.matmul(layer.U_additional.cuda(), layer.Vh_additional.cuda())
    ).data
    U, S, Vh = torch.linalg.svd(weight)
    rank = rank_cal(S)
    U = U[:, :rank]
    Vh = torch.matmul(torch.diag(S[:rank]), Vh[:rank, :])
    first_layer = torch.nn.Linear(
        in_features=Vh.shape[1], out_features=Vh.shape[0], bias=False
    )
    second_layer = torch.nn.Linear(
        in_features=U.shape[1], out_features=U.shape[0], bias=bias_flag
    )
    if bias_flag:
        second_layer.bias.data = layer.bias.data.clone().cpu()
    first_layer.weight.data = Vh.clone().cpu()
    second_layer.weight.data = U.clone().cpu()
    new_layers = [first_layer, second_layer]
    return nn.Sequential(*new_layers)


def select_basis(model, threshold) -> None:
    for _, layer in model._modules.items():
        if len(list(layer.children())) > 0:
            select_basis(layer, threshold)
        else:
            if type(layer) == BSLinear:
                select_basis_layer(layer, threshold)
        torch.cuda.empty_cache()


def select_basis_layer(layer, threshold) -> None:
    weight = torch.sort((layer.weight**2) * layer.mask, descending=True).values
    partial_sum = torch.cumsum(weight, dim=0)
    indices = torch.nonzero(partial_sum >= (threshold * torch.sum(weight)))
    if indices.numel() > 0:
        rank = indices[0, 0] + 1
    else:
        rank = weight.size(0)
    layer.mask = torch.where(
        (layer.weight**2) * layer.mask >= weight[rank - 1], 1.0, 0.0
    )


def decompress_state(model, cpu_state, prev_name=None):
    for name, module in model._modules.items():
        if prev_name is not None:
            name = prev_name + "." + name

        if isinstance(module, nn.Sequential):
            if (len(module) == 2 and
                isinstance(module[0], nn.Linear) and
                isinstance(module[1], nn.Linear)):

                # Merge the two factorized weights
                w1 = cpu_state[name + '.1.weight'].cuda()
                w0 = cpu_state[name + '.0.weight'].cuda()
                cpu_state[name + '.weight'] = torch.mm(w1, w0).cpu()

                # Cleanup
                del cpu_state[name + '.0.weight'], cpu_state[name + '.1.weight']
                del w1, w0
                torch.cuda.empty_cache()

            else:
                decompress_state(module, cpu_state, name)

        else:
            decompress_state(module, cpu_state, name)

    torch.cuda.empty_cache()


def unwrap_fsdp(model):
        for name, module in model.named_children():
            if isinstance(module, FullyShardedDataParallel):
                model._modules[name] = module.module
            else:
                model._modules[name] = unwrap_fsdp(module)

        return model


def output_dim_json(cpu_state, path) -> None:
    def get_prefix(s):
        if s.endswith(".0.weight") or s.endswith(".1.weight"):
            return s.rsplit(".", 1)[0]
        else:
            return s

    dim_dict = {}
    for k in cpu_state.keys():

        if k == "model.embed_tokens.weight":
            continue
        dim = cpu_state[k].shape
        if len(dim) == 2:
            dim_dict[get_prefix(k)] = list(dim)
    with open(os.path.join(path, "dim.json"), "w") as file:
        json.dump(dim_dict, file)


