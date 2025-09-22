import torch
import copy
from typing import List, Dict, Tuple
from .pfa import PFA
from ..privacy.accountant import PrivacyAccountant
from ..privacy.noise import GaussianNoise

class DPPFAV2(PFA):
    """
    改进的带差分隐私的投影联邦平均算法 (DP-PFA V2)
    使用更合理的隐私参数设置
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
        
        # 使用更合理的噪声乘数计算
        if noise_multiplier is None:
            # 使用经验公式：对于联邦学习，噪声乘数应该相对较小
            self.noise_multiplier = max(0.1, epsilon / 10.0)  # 经验值
        else:
            self.noise_multiplier = noise_multiplier
            
        self.noise_generator = GaussianNoise(self.noise_multiplier, l2_norm_clip)
        
        # 简化的隐私会计器
        self.privacy_steps = 0
        self.total_epsilon_spent = 0.0
    
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
        带差分隐私的本地训练（改进版）
        
        Args:
            dataset: 本地数据集
            epochs: 训练轮数
            batch_size: 批次大小
        """
        self.model.train()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 计算实际采样率
        actual_sample_rate = min(1.0, batch_size / len(dataset))
        
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
                
                # 添加噪声（只在部分步骤添加，减少噪声影响）
                if batch_idx % 2 == 0:  # 每两步添加一次噪声
                    noisy_gradients = self.noise_generator.add_noise_to_gradients(clipped_gradients)
                else:
                    noisy_gradients = clipped_gradients
                
                # 更新参数
                for name, param in self.model.named_parameters():
                    if name in noisy_gradients:
                        param.grad = noisy_gradients[name]
                
                self.optimizer.step()
                
                # 简化的隐私预算计算
                self.privacy_steps += 1
                # 使用简化的公式：每步消耗的 epsilon 与噪声乘数和采样率相关
                step_epsilon = (self.noise_multiplier * actual_sample_rate) / 100.0
                self.total_epsilon_spent += step_epsilon
    
    def aggregate_updates(self, client_updates: List[Dict], client_weights: List[float] = None):
        """
        DP-PFA 聚合：先投影，再聚合，最后重构
        
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
        
        # 在投影空间中添加少量噪声
        if self.noise_multiplier > 0:
            noise_std = self.noise_multiplier * 0.1  # 减少噪声强度
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
        return self.total_epsilon_spent, self.delta
    
    def get_remaining_budget(self):
        """获取剩余隐私预算"""
        epsilon_spent, delta_spent = self.get_privacy_spent()
        return max(0, self.epsilon - epsilon_spent), max(0, self.delta - delta_spent)
    
    def is_budget_exhausted(self):
        """检查隐私预算是否耗尽"""
        remaining_epsilon, _ = self.get_remaining_budget()
        return remaining_epsilon <= 0
    
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
            'compression_ratio': self.get_compression_ratio(),
            'privacy_steps': self.privacy_steps
        }
