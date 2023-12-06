import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import utils
from spatial_transformer import STN

class PointNet(nn.Module):
    def __init__(self, n_scales = 1, n_points = 500, use_point_stn = True, use_feat_stn = True, sym_op = 'max', get_pointfvals = False, point_tuple = 1) -> None:
        super(PointNet, self).__init__()
        self.n_points = n_points
        self.n_scales = n_scales
        self.use_point_stn = True
        self.use_feat_stn = True
        self.sym_op = sym_op
        self.get_pointfvals = get_pointfvals
        self.point_tuple = point_tuple

        if self.use_point_stn:
            self.stn1 = STN(n_scales= self.n_scales, 
                            n_points= self.n_points*self.point_tuple,
                            n_dim=3,
                            quaternion= True)
        if self.use_feat_stn:
            self.stn2 = STN(n_scales= self.n_scales,
                            n_points= self.n_points,
                            n_dim= 64,
                            quaternion= False)
        self.conv0a = nn.Conv1d(3*self.point_tuple, 64, 1)
        self.conv0b = nn.Conv1d(64, 64, 1)
        self.bn0a = nn.BatchNorm1d(64)
        self.bn0b = nn.BatchNorm1d(64)
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.n_scales > 1:
            self.conv4 = nn.Conv1d(1024, 1024*self.n_scales, 1)
            self.bn4 = nn.BatchNorm1d(1024*self.n_scales)

        if self.sym_op == 'max':
            self.max_pool = nn.MaxPool1d(self.n_points)
        else:
            raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
        
    def forward(self, x):

        if self.use_point_stn:
            x = x.view(x.size(0), 3, -1)
            trans1 = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans1)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3*self.point_tuple, -1)
        else:
            trans1 = None
        
        x = F.relu(self.bn0a(self.conv0a(x)))
        x = F.relu(self.bn0b(self.conv0b(x)))

        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.n_scales > 1:
            x = self.bn4(self.conv4(F.relu(x)))
        if self.get_pointfvals:
            pointfvals = x
        else:
            pointfvals = None
        
        if self.n_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales**2, 1)
            if self.sym_op == 'max':
                for s in range(self.n_scales):
                    x_scales[:, s*self.n_scales*1024:(s+1)*self.n_scales*1024, :] = self.max_pool(x[:, :, s*self.n_points:(s+1)*self.n_points])
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
            x = x_scales
        
        x = x.view(-1, 1024*self.n_scales**2)

        return x, trans1, trans2, pointfvals
    
class PCPNet(nn.Module):
    def __init__(self, n_points = 500, out_dim = 3, use_point_stn = True, use_feat_stn = True, sym_op = 'max', get_pointfvals = False, point_tuple = 1) -> None:
        super(PCPNet, self).__init__()
        self.n_points = n_points
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.get_pointfvals = get_pointfvals
        self.point_tuple = point_tuple
        self.feat = PointNet(n_points= self.n_points,
                             n_scales= 1, 
                             use_feat_stn= self.use_feat_stn,
                             use_point_stn= self.use_point_stn,
                             sym_op= self.sym_op,
                             get_pointfvals= self.get_pointfvals,
                             point_tuple= self.point_tuple)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop_out1 = nn.Dropout(p=0.3)
        self.drop_out2 = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x, trans1, trans2, pointfvals = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop_out1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop_out2(x)
        x = self.fc3(x)

        return x, trans1, trans2, pointfvals

