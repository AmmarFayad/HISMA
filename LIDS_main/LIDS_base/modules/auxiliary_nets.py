from numpy import add
import torch
import torch.optim as optim

from torch import nn as nn
from torch.nn import functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Err(nn.Module):

    def __init__(self, num_inputs, z_dim,embed_dim, num_outputs, lr=3e-4):
        super(Err, self).__init__()

        self.linear1 = nn.Linear(num_inputs + z_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.last_fc = nn.Linear(embed_dim, num_outputs)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        h = F.relu(self.linear1(input))
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x


    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

            return loss.to('cpu').detach().item()

        return None


class Diff(nn.Module):

    def __init__(self, num_inputs, z_dim,embed_dim, num_outputs, lr=3e-4):
        super(Err, self).__init__()
        

        self.linear1 = nn.Linear(num_inputs , embed_dim)
        self.linear1z = nn.Linear(z_dim, embed_dim)
        self.linear2z = nn.Linear(z_dim, embed_dim)
        self.conv=nn.Conv1d(num_inputs,embed_dim)
        self.linear2 = nn.Linear(num_inputs, embed_dim)
        self.linearall = nn.Linear(embed_dim, embed_dim)
        self.last_fc = nn.Linear(embed_dim, num_outputs)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input,z):
        h = torch.mul(F.sigmoid(self.conv(self.linear1(input))+self.linear1z(z)),F.tanh(self.conv(self.linear2(input))+self.linear2z(z)))
        h = F.relu(self.linearall(h))
        x = self.last_fc(h)
        return x


    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

            return loss.to('cpu').detach().item()

        return None