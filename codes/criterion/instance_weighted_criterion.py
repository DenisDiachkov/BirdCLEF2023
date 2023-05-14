import torch
import torch.nn as nn
import utils


class InstanceWeightedCriterion(nn.Module):
    def __init__(self, criterion):
        super(InstanceWeightedCriterion, self).__init__()
        if isinstance(criterion, dict):
            self.criterion = utils.get_instance(criterion.criterion, criterion.criterion_params)
        elif isinstance(criterion, str):
            self.criterion = utils.get_instance(criterion)
        elif isinstance(criterion, torch.nn.Module):
            self.criterion = criterion
    
    def forward(self, x, y, weight):
        return self.criterion(x, y) * weight
    