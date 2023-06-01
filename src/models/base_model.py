import torch.nn as nn

class BaseModel(nn.Module):
    def get_learned_features(self):
        raise NotImplementedError