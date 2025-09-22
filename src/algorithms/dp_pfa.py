import torch
import copy
from typing import List, Dict, Tuple
from .pfa import PFA
from ..privacy.accountant import PrivacyAccountant
from ..privacy.noise import GaussianNoise

class DPPFA(PFA):
    """
    带差分隐私的投影联邦平均算法 (DP-PFA)
    """
    
    def __init__(self, model, lr=0.1, proj_dims=1, lanczos_iter=64, device='cpu',
                 epsilon=1.0, delta=1e-5, noise_multiplier=None,
                 l2_norm_clip=1.0, sample_rate=1.0):
        super().__init__(model, lr, proj_dims, lanczos_iter, device)
        
        # 差分隐私参数
        self.epsilon = epsilon
        self.delta = delta
        self.l2_norm_clip = l2_norm_clip
        self.sample_rate = sample_rate
        
        # 初始化隐私会计器
        self.privacy_accountant = PrivacyAccountant(epsilon, delta, noise_multiplier)
        
        # 如果没有指定噪声乘数，计算一个
        if noise_multiplier is None:
            self.noise_multiplier = self.privacy_accountant.get_noise_multiplier_for_target_epsilon(
                epsilon, sample_rate, 100  # 假设100步训练
            )
        else:
            self.noise_multiplier = noise_multiplier
            
        self.noise_generator = GaussianNoise(self.noise_multiplier, l2_norm_clip)
    
    def clip_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        裁剪梯度到 L2 范数
        
        Args:
            gradients: 梯度字典
            
        Returns:
            clipped_gradients: 裁剪后的梯度
        """
        # 计算总梯度范数
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # 裁剪系数
        clip_coef = min(1.0, self.l2_norm_clip / (total_norm + 1e-6))
        
        # 应用裁剪
        clipped_gradients = {}
        for key, grad in gradients.items():
            clipped_gradients[key] = grad * clip_coef
            
        return clipped_gradients
    
    def local_update(self, dataset, epochs=1, batch_size=32):
        """
        带差分隐私的本地训练
        
        Args:
            dataset: 本地数据集
            epochs: 训练轮数
            batch_size: 批次大小
        """
        self.model.train()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # 反向传播
                loss.backward()
                
                # 获取梯度
                gradients = {}
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.clone()
                
                # 裁剪梯度
                clipped_gradients = self.clip_gradients(gradients)
                
                # 添加噪声
                noisy_gradients = self.noise_generator.add_noise_to_gradients(clipped_gradients)
                
                # 更新参数
                for name, param in self.model.named_parameters():
                    if name in noisy_gradients:
                        param.grad = noisy_gradients[name]
                
                self.optimizer.step()
                
                # 记录隐私预算消耗
                self.privacy_accountant.step(self.noise_multiplier, self.sample_rate)
    
    def aggregate_updates(self, client_updates: List[Dict], client_weights: List[float] = None):
        """
        DP-PFA 聚合：先投影，再聚合，最后重构，整个过程都考虑差分隐私
        
        Args:
            client_updates: 客户端更新列表
            client_weights: 客户端权重列表
        """
        if not client_updates:
            return
        
        # 保存当前全局模型状态
        self.global_model_state = copy.deepcopy(self.get_model_state())
        
        # 计算投影
        projection_matrix, projected_updates = self.compute_projection(client_updates)
        
        # 在投影空间中聚合
        if client_weights is None:
            client_weights = [1.0 / len(projected_updates)] * len(projected_updates)
        
        aggregated_projected = torch.zeros_like(projected_updates[0])
        for i, proj_update in enumerate(projected_updates):
            aggregated_projected += client_weights[i] * proj_update
        
        # 在投影空间中添加噪声（可选）
        if self.noise_multiplier > 0:
            noise_std = self.noise_multiplier * self.l2_norm_clip
            noise = torch.randn_like(aggregated_projected) * noise_std
            aggregated_projected += noise
        
        # 重构到原始空间
        aggregated_update_vector = projection_matrix @ aggregated_projected
        
        # 将更新向量转换回模型状态字典
        aggregated_update_dict = self._unflatten_model_state(
            aggregated_update_vector, self.global_model_state
        )
        
        # 应用更新到全局模型
        new_global_state = {}
        for key in self.global_model_state.keys():
            if key in aggregated_update_dict:
                new_global_state[key] = self.global_model_state[key] + aggregated_update_dict[key]
            else:
                new_global_state[key] = self.global_model_state[key]
        
        self.set_model_state(new_global_state)
    
    def _unflatten_model_state(self, flattened: torch.Tensor, model_state_template: dict) -> dict:
        """
        将展平的向量重构为模型状态字典
        
        Args:
            flattened: 展平的向量
            model_state_template: 模型状态模板
            
        Returns:
            model_state: 重构的模型状态字典
        """
        model_state = {}
        start_idx = 0
        
        for key, template_tensor in model_state_template.items():
            end_idx = start_idx + template_tensor.numel()
            tensor_data = flattened[start_idx:end_idx]
            model_state[key] = tensor_data.reshape(template_tensor.shape)
            start_idx = end_idx
        
        return model_state
    
    def get_privacy_spent(self):
        """获取已消耗的隐私预算"""
        return self.privacy_accountant.get_privacy_spent()
    
    def get_remaining_budget(self):
        """获取剩余隐私预算"""
        return self.privacy_accountant.get_remaining_budget()
    
    def is_budget_exhausted(self):
        """检查隐私预算是否耗尽"""
        return self.privacy_accountant.is_budget_exhausted()
    
    def get_privacy_info(self):
        """获取隐私信息摘要"""
        epsilon_spent, delta_spent = self.get_privacy_spent()
        remaining_epsilon, remaining_delta = self.get_remaining_budget()
        
        return {
            'epsilon_spent': epsilon_spent,
            'delta_spent': delta_spent,
            'remaining_epsilon': remaining_epsilon,
            'remaining_delta': remaining_delta,
            'noise_multiplier': self.noise_multiplier,
            'l2_norm_clip': self.l2_norm_clip,
            'compression_ratio': self.get_compression_ratio()
        }
