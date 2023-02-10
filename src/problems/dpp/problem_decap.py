from torch.utils.data import Dataset
import torch
import os
import pickle
from tqdm import tqdm

from src.problems.dpp.state_decap import StateDecap
from src.problems.dpp.reward_function_serial import reward_gen


class Decap(object):
    NAME = "dpp"

    @staticmethod
    def get_costs(dataset, pi, raw_pdn=None, z_init_list=None, mask=None):
        # Check that tours are valid, i.e. contain 0 to n -1

        # Gather dataset in order of tour

        reward_list = []

        for i in tqdm(range(pi.size(0))):
            #             trial[0] = pi[i].cpu().numpy()

            reward = reward_gen(
                (dataset[i, :, 2] == 2).nonzero().item(), pi[i].cpu().numpy(), 5
            )

            reward_list.append(reward)

        # reward = reward_gen((dataset[:,:,2]==2).nonzero()[:,0], pi.cpu().numpy(),5, raw_pdn,z_init_list,mask)
        #         assert(False)
        #         reward_list.append(reward)

        reward = torch.FloatTensor(reward_list).cuda().view(-1, 1)

        # virtual reward
        # d = dataset.gather(1, pi.unsqueeze(-1))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return -reward, None
        # return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return DecapDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateDecap.initialize(*args, **kwargs)


class DecapDataset(Dataset):
    def __init__(
        self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None
    ):
        super(DecapDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == ".pkl"

            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.data = [
                    torch.FloatTensor(row)
                    for row in (data[offset : offset + num_samples])
                ]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [
                torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
