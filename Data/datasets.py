import torch
from torch.utils.data import Dataset
from mineclip.mineagent.batch import Batch

class PPODataset(Dataset):
    def __init__(self, data: dict, device: torch.device) -> None:
        self.state_data = data['states']
        self.action_data = data['actions'].to(device)
        self.advantage_data = data['advantages'].to(device)
        self.old_logp_data = data['log_probs'].to(device)
        self.return_data = data['returns'].to(device)
    
    def __len__(self):
        return len(self.action_data)
    
    def __getitem__(self, index):
        
        state = self.state_data[index]
        action = self.action_data[index]
        advantage = self.advantage_data[index]
        old_logp = self.old_logp_data[index]
        retur = self.return_data[index]

        return state, action, advantage, old_logp, retur

def custom_collate_fn(batch):
    arranged_batch = list(zip(*batch))
    state_batch = Batch.stack(arranged_batch[0])
    action_batch = torch.stack(arranged_batch[1])
    advantage_batch = torch.stack(arranged_batch[2])
    old_logp_batch = torch.stack(arranged_batch[3])
    return_batch = torch.stack(arranged_batch[4])

    return state_batch, action_batch, advantage_batch, old_logp_batch, return_batch