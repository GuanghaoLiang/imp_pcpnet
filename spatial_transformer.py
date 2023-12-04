import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
class STN(nn.Module):
    def __init__(self, n_dim = 3, n_scales = 1, n_points = 500, sym_op = 'max', quaternion = False) -> None:
        super(STN, self).__init__()
        self.n_dim = n_dim
        self.n_scales = n_scales
        self.sym_op = sym_op
        self.n_points = n_points
        self.quaternion = quaternion

        self.conv1 = nn.Conv1d(self.n_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.max_pool = nn.MaxPool1d(self.n_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.n_dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.n_scales > 1:
            self.fc0 = nn.Linear(1024*self.n_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)
    
    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        if self.n_scales == 1:
            x = self.max_pool(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.n_scales, 1)
            for s in range(self.n_scales):
                x_scales[:, s*1024:s*1024+1024, :] = self.max_pool(x[:, :, s*self.n_points:(s+1)*self.n_points])
            x = x_scales
        
        x = x.view(-1, 1024*self.n_scales)

        if self.n_scales>1:
            x = F.relu(self.bn0(self.fc0(x)))
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        if self.quaternion:
            idx = x.new_tensor([1, 0, 0, 0])
            x = utils.batch_quat_to_rotmat(x)
        else:
            idx = torch.eye(self.n_dim, dtype=x.dtype, device=x.device).view(1, self.n_dim*self.n_dim).repeat(batch_size, 1)
            x = x + idx
            x = x.view(-1, self.n_dim, self.n_dim)
        
        return x


        
