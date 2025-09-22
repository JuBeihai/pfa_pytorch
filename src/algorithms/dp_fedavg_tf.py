import torch
import copy
import math
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader
from .fedavg import FedAvg
from ..privacy.accountant import PrivacyAccountant

class DPFedAvg_TF(FedAvg):
    """
    带差分隐私的联邦平均算法，完全匹配 TensorFlow 版本
    """
    def __init__(self, model, lr=0.1, device='cpu',
                 epsilon: float = 1.0, delta: float = 1e-5, 
                 noise_multiplier: Optional[float] = None, 
                 l2_norm_clip: float = 1.0, sample_rate: float = 1.0):
        
        super().__init__(model, lr, device)
        self.epsilon = epsilon
        self.delta = delta
        self.l2_norm_clip = l2_norm_clip
        self.sample_rate = sample_rate
        
        # 使用 TensorFlow 版本的噪声乘数计算
        if noise_multiplier is None:
            # 需要根据实际参数计算
            self.noise_multiplier = None  # 将在训练时计算
        else:
            self.noise_multiplier = noise_multiplier
        
        self.privacy_accountant = PrivacyAccountant(epsilon, delta, self.noise_multiplier)
        
    def local_update(self, dataset, local_steps=100, batch_size=4):
        """
        客户端本地训练，匹配 TensorFlow 版本的 DP 实现
        """
        self.model.train()
        
        # 获取数据集大小
        dataset_size = len(dataset)
        
        # 计算噪声乘数（如果未设置）
        if self.noise_multiplier is None:
            # 使用 TensorFlow 版本的公式
            N = dataset_size  # 客户端数据集大小
            L = batch_size    # 批次大小
            T = local_steps   # 本地步数
            self.noise_multiplier = self.privacy_accountant.compute_noise_multiplier(
                N, L, T, self.epsilon, self.delta
            )
            # 更新 privacy_accountant 的噪声乘数
            self.privacy_accountant.noise_multiplier = self.noise_multiplier
        
        for step in range(local_steps):
            # 随机采样一个 batch，匹配 TensorFlow 版本
            batch_indices = torch.randperm(dataset_size)[:batch_size]
            
            # 获取 batch 数据
            batch_data = []
            batch_targets = []
            for idx in batch_indices:
                data, target = dataset[idx]
                batch_data.append(data)
                batch_targets.append(target)
            
            data = torch.stack(batch_data).to(self.device)
            target = torch.stack(batch_targets).to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # 应用 DP 梯度裁剪和噪声
            self._apply_dp_to_gradients()
            
            self.optimizer.step()
        
        # 更新隐私预算
        self.privacy_accountant.update(local_steps)
        
        return self.get_model_state()
    
    def _apply_dp_to_gradients(self):
        """
        对梯度应用差分隐私，匹配 TensorFlow 版本的实现
        """
        # 1. 梯度裁剪 (L2 norm clipping)
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = self.l2_norm_clip / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)

        # 2. 添加高斯噪声
        if self.noise_multiplier > 0:
            for p in self.model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad.data) * self.l2_norm_clip * self.noise_multiplier
                    p.grad.data.add_(noise)
    
    def precheck(self, dataset_size: int, batch_size: int, loc_steps: int) -> bool:
        """
        预检查客户端是否可以参与下一轮训练
        """
        return self.privacy_accountant.precheck(dataset_size, batch_size, loc_steps)
    
    def get_privacy_info(self) -> Dict[str, float]:
        """
        获取隐私信息
        """
        epsilon_spent, delta_spent = self.privacy_accountant.get_privacy_spent()
        remaining_epsilon, remaining_delta = self.privacy_accountant.get_remaining_budget()
        return {
            "epsilon_spent": epsilon_spent,
            "delta_spent": delta_spent,
            "remaining_epsilon": remaining_epsilon,
            "remaining_delta": remaining_delta,
            "noise_multiplier": self.noise_multiplier,
            "l2_norm_clip": self.l2_norm_clip,
            "sample_rate": self.sample_rate,
        }
    
    def is_budget_exhausted(self) -> bool:
        """
        检查隐私预算是否耗尽
        """
        return self.privacy_accountant.is_budget_exhausted()
