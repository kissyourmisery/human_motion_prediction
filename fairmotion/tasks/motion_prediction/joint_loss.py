import torch
import torch.nn as nn


class PerJointLoss(nn.Module):
    def __init__(self, n_joints=24, joint_dim=3, timesteps=120, device='cpu'):
        super(PerJointLoss, self).__init__()
        self.n_joints = n_joints
        self.joint_dim = joint_dim
        self.timesteps = timesteps
        self.device = device

    def forward(self, src, tgt):
        n = 0
        t = 0
        sum_n = torch.tensor(0.0).to(self.device)
        sum_t = torch.tensor(0.0).to(self.device)
        while n < self.n_joints and t < self.timesteps:
            j_t_n = src[:, self.joint_dim * n: self.joint_dim * (n + 1), t]
            j_t_n_p = tgt[:, self.joint_dim * n: self.joint_dim * (n + 1), t]
            sum_n += torch.linalg.norm(j_t_n - j_t_n_p)
            if n == self.n_joints - 1:
                t += 1
                n = 0
                sum_t += sum_n
                sum_n = torch.tensor(0.0).to(self.device)
            else:
                n += 1
        return sum_t
