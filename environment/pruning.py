import torch
import torch.nn as nn


def apply_mask(model, mask):
    with torch.no_grad():
        for name, param in mask.items():
            name_ = name.replace('-', '.') # Changing to the original key
            model.state_dict()[name_].data.copy_( model.state_dict()[name_].data.mul(param.data) )


def create_mask(model): # Create mask as Lottery Tickets Hypothesis
    from collections import OrderedDict
    mask = OrderedDict()
    for name, param in model.named_parameters():
        if 'conv' in name and 'bias' not in name:
            name_ = name.replace('.', '-') # ParameterDict and ModuleDict does not allows '.' as key
            mask[name_] = nn.Parameter( torch.ones_like(param), requires_grad = False )

    return nn.ParameterDict(mask)


def prune_mask(mask, index):
    mask.weight[index] = nn.Parameter( torch.zeros_like(mask[index]), requires_grad=False )


def unprune_mask(mask, index):
    mask.weight[index] = nn.Parameter( torch.ones_like(mask[index]), requires_grad=False )