__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch.nn as nn

class CELoss(nn.Module):
    def __init__(self,
                 reduction):
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, preds, target):
        return self.ce(preds, target.squeeze(2))


