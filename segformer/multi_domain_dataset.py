import os
from torch.utils.data import Dataset
from dataset import SegmentationDataset

class MultiDomainDataset(Dataset):
    def __init__(self, root_dir, domains, augment=False):
        self.datasets = []
        for domain in domains:
            domain_path = os.path.join(root_dir, domain)
            self.datasets.append(
                SegmentationDataset(domain_path, augment=augment)
            )

        self.lengths = [len(d) for d in self.datasets]
        self.cum_lengths = []
        total = 0
        for l in self.lengths:
            total += l
            self.cum_lengths.append(total)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        for i, c in enumerate(self.cum_lengths):
            if idx < c:
                prev = 0 if i == 0 else self.cum_lengths[i - 1]
                return self.datasets[i][idx - prev]