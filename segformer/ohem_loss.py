import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMLoss(nn.Module):

    def __init__(self, thresh=0.7, min_kept=10000):

        super().__init__()

        self.thresh = thresh
        self.min_kept = min_kept

        self.criteria = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, target):

        loss = self.criteria(logits, target)

        loss = loss.view(-1)

        sorted_loss, _ = torch.sort(loss, descending=True)

        if sorted_loss[self.min_kept] > self.thresh:
            threshold = sorted_loss[self.min_kept]
        else:
            threshold = self.thresh

        hard_loss = loss[loss > threshold]

        return torch.mean(hard_loss)