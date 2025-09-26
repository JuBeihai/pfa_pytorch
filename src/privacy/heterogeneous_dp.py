"""
异构差分隐私模块
完全匹配论文中的异构差分隐私实现，支持每个客户端不同的隐私参数
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
from .accountant import PrivacyAccountant

class HeterogeneousDP:
    """
    异构差分隐私实现
    支持每个客户端不同的隐私参数{(εm, δm)}m∈[M]
    """
    
    def __init__(self, client_epsilons: List[float], 
                 client_deltas: List[float],
                 global_epsilon: Optional[float] = None,
                 global_delta: Optional[float] = None):
        """
        初始化异构差分隐私
        
        Args:
            client_epsilons: 每个客户端的ε参数
            client_deltas: 每个客户端的δ参数
            global_epsilon: 全局ε参数（用于整体DP保证）
            global_delta: 全局δ参数（用于整体DP保证）
        """
        self.client_epsilons = np.array(client_epsilons)
        self.client_deltas = np.array(client_deltas)
        self.n_clients = len(client_epsilons)
        
        # 验证参数
        self._validate_parameters()
        
        # 计算全局隐私参数
        if global_epsilon is None:
            self.global_epsilon = np.max(self.client_epsilons)
        else:
            self.global_epsilon = global_epsilon
            
        if global_delta is None:
            self.global_delta = np.max(self.client_deltas)
        else:
            self.global_delta = global_delta
        
        # 为每个客户端创建隐私会计器
        self.client_accountants = []
        for i in range(self.n_clients):
            accountant = PrivacyAccountant(
                epsilon=self.client_epsilons[i],
                delta=self.client_deltas[i]
            )
            self.client_accountants.append(accountant)
        
        # 全局隐私会计器
        self.global_accountant = PrivacyAccountant(
            epsilon=self.global_epsilon,
            delta=self.global_delta
        )
        
        # 隐私预算跟踪
        self.budget_tracking = {
            'client_epsilons_spent': [0.0] * self.n_clients,
            'client_deltas_spent': [0.0] * self.n_clients,
            'global_epsilon_spent': 0.0,
            'global_delta_spent': 0.0
        }
    
    def _validate_parameters(self):
        """
        验证隐私参数
        """
        if len(self.client_epsilons) != len(self.client_deltas):
            raise ValueError("客户端ε和δ参数数量不匹配")
        
        if np.any(self.client_epsilons <= 0):
            raise ValueError("所有ε参数必须大于0")
        
        if np.any(self.client_deltas <= 0) or np.any(self.client_deltas >= 1):
            raise ValueError("所有δ参数必须在(0,1)范围内")
    
    def compute_noise_multipliers(self, client_dataset_sizes: List[int], 
                                 batch_sizes: List[int], 
                                 local_steps: List[int]) -> List[float]:
        """
        为每个客户端计算噪声乘数
        
        Args:
            client_dataset_sizes: 客户端数据集大小
            batch_sizes: 客户端批次大小
            local_steps: 客户端本地训练步数
            
        Returns:
            noise_multipliers: 每个客户端的噪声乘数
        """
        noise_multipliers = []
        
        for i in range(self.n_clients):
            # 计算采样率
            q = batch_sizes[i] / client_dataset_sizes[i]
            
            # 使用RDP会计器计算噪声乘数
            noise_mult = self.client_accountants[i].compute_noise_multiplier(
                N=client_dataset_sizes[i],
                L=batch_sizes[i],
                T=local_steps[i],
                epsilon=self.client_epsilons[i],
                delta=self.client_deltas[i]
            )
            
            noise_multipliers.append(noise_mult)
        
        return noise_multipliers
    
    def apply_heterogeneous_dp(self, client_gradients: List[Dict[str, torch.Tensor]], 
                             client_ids: List[int],
                             noise_multipliers: List[float],
                             l2_norm_clip: float = 1.0) -> List[Dict[str, torch.Tensor]]:
        """
        对客户端梯度应用异构差分隐私
        
        Args:
            client_gradients: 客户端梯度列表
            client_ids: 客户端ID列表
            noise_multipliers: 噪声乘数列表
            l2_norm_clip: L2范数裁剪参数
            
        Returns:
            dp_gradients: 应用DP后的梯度列表
        """
        dp_gradients = []
        
        for i, (gradients, client_id, noise_mult) in enumerate(
            zip(client_gradients, client_ids, noise_multipliers)
        ):
            # 应用L2范数裁剪
            clipped_gradients = self._clip_gradients(gradients, l2_norm_clip)
            
            # 添加高斯噪声
            noisy_gradients = self._add_gaussian_noise(
                clipped_gradients, noise_mult, l2_norm_clip
            )
            
            dp_gradients.append(noisy_gradients)
            
            # 更新隐私预算
            self._update_client_budget(client_id, 1)  # 假设每步消耗1个预算单位
        
        return dp_gradients
    
    def _clip_gradients(self, gradients: Dict[str, torch.Tensor], 
                       l2_norm_clip: float) -> Dict[str, torch.Tensor]:
        """
        对梯度应用L2范数裁剪
        """
        # 计算总梯度范数
        total_norm = 0.0
        for grad in gradients.values():
            if grad is not None:
                total_norm += grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # 计算裁剪系数
        clip_coef = min(1.0, l2_norm_clip / (total_norm + 1e-8))
        
        # 应用裁剪
        clipped_gradients = {}
        for key, grad in gradients.items():
            if grad is not None:
                clipped_gradients[key] = grad * clip_coef
            else:
                clipped_gradients[key] = grad
        
        return clipped_gradients
    
    def _add_gaussian_noise(self, gradients: Dict[str, torch.Tensor], 
                           noise_multiplier: float, 
                           l2_norm_clip: float) -> Dict[str, torch.Tensor]:
        """
        添加高斯噪声
        """
        noisy_gradients = {}
        
        for key, grad in gradients.items():
            if grad is not None:
                # 计算噪声标准差
                noise_std = l2_norm_clip * noise_multiplier
                
                # 生成高斯噪声
                noise = torch.randn_like(grad) * noise_std
                
                # 添加噪声
                noisy_gradients[key] = grad + noise
            else:
                noisy_gradients[key] = grad
        
        return noisy_gradients
    
    def _update_client_budget(self, client_id: int, steps: int):
        """
        更新客户端隐私预算
        """
        if 0 <= client_id < self.n_clients:
            # 更新客户端预算
            self.client_accountants[client_id].update(steps)
            
            # 更新预算跟踪
            epsilon_spent, delta_spent = self.client_accountants[client_id].get_privacy_spent()
            self.budget_tracking['client_epsilons_spent'][client_id] = epsilon_spent
            self.budget_tracking['client_deltas_spent'][client_id] = delta_spent
    
    def update_global_budget(self, steps: int):
        """
        更新全局隐私预算
        """
        self.global_accountant.update(steps)
        epsilon_spent, delta_spent = self.global_accountant.get_privacy_spent()
        self.budget_tracking['global_epsilon_spent'] = epsilon_spent
        self.budget_tracking['global_delta_spent'] = delta_spent
    
    def check_client_budget(self, client_id: int) -> bool:
        """
        检查客户端隐私预算是否足够
        """
        if 0 <= client_id < self.n_clients:
            return not self.client_accountants[client_id].is_budget_exhausted()
        return False
    
    def check_global_budget(self) -> bool:
        """
        检查全局隐私预算是否足够
        """
        return not self.global_accountant.is_budget_exhausted()
    
    def get_client_privacy_info(self, client_id: int) -> Dict[str, float]:
        """
        获取客户端隐私信息
        """
        if 0 <= client_id < self.n_clients:
            epsilon_spent, delta_spent = self.client_accountants[client_id].get_privacy_spent()
            remaining_epsilon, remaining_delta = self.client_accountants[client_id].get_remaining_budget()
            
            return {
                'client_id': client_id,
                'epsilon_original': self.client_epsilons[client_id],
                'delta_original': self.client_deltas[client_id],
                'epsilon_spent': epsilon_spent,
                'delta_spent': delta_spent,
                'remaining_epsilon': remaining_epsilon,
                'remaining_delta': remaining_delta,
                'budget_exhausted': self.client_accountants[client_id].is_budget_exhausted()
            }
        else:
            raise ValueError(f"无效的客户端ID: {client_id}")
    
    def get_global_privacy_info(self) -> Dict[str, float]:
        """
        获取全局隐私信息
        """
        epsilon_spent, delta_spent = self.global_accountant.get_privacy_spent()
        remaining_epsilon, remaining_delta = self.global_accountant.get_remaining_budget()
        
        return {
            'global_epsilon_original': self.global_epsilon,
            'global_delta_original': self.global_delta,
            'global_epsilon_spent': epsilon_spent,
            'global_delta_spent': delta_spent,
            'remaining_global_epsilon': remaining_epsilon,
            'remaining_global_delta': remaining_delta,
            'global_budget_exhausted': self.global_accountant.is_budget_exhausted()
        }
    
    def get_heterogeneous_privacy_guarantee(self) -> Dict[str, any]:
        """
        获取异构差分隐私保证
        """
        # 计算每个客户端的隐私保证
        client_guarantees = []
        for i in range(self.n_clients):
            client_info = self.get_client_privacy_info(i)
            client_guarantees.append({
                'client_id': i,
                'epsilon': client_info['epsilon_original'],
                'delta': client_info['delta_original'],
                'epsilon_spent': client_info['epsilon_spent'],
                'delta_spent': client_info['delta_spent']
            })
        
        # 计算全局隐私保证
        global_info = self.get_global_privacy_info()
        
        return {
            'client_guarantees': client_guarantees,
            'global_guarantee': {
                'epsilon': global_info['global_epsilon_original'],
                'delta': global_info['global_delta_original'],
                'epsilon_spent': global_info['global_epsilon_spent'],
                'delta_spent': global_info['global_delta_spent']
            },
            'satisfies_fl_hdp': all(
                not self.client_accountants[i].is_budget_exhausted() 
                for i in range(self.n_clients)
            ),
            'satisfies_global_dp': not self.global_accountant.is_budget_exhausted()
        }
    
    def reset_budgets(self):
        """
        重置所有隐私预算
        """
        for accountant in self.client_accountants:
            accountant.reset()
        self.global_accountant.reset()
        
        # 重置预算跟踪
        self.budget_tracking = {
            'client_epsilons_spent': [0.0] * self.n_clients,
            'client_deltas_spent': [0.0] * self.n_clients,
            'global_epsilon_spent': 0.0,
            'global_delta_spent': 0.0
        }
