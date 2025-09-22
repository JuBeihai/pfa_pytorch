import torch
import numpy as np
from typing import Dict, List, Tuple
from .accountant import PrivacyAccountant

class GaussianNoise:
    """
    高斯噪声生成器，用于差分隐私
    """
    
    def __init__(self, noise_multiplier: float, l2_norm_clip: float = 1.0):
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        
    def add_noise_to_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        向梯度添加高斯噪声
        
        Args:
            gradients: 梯度字典
            
        Returns:
            noisy_gradients: 添加噪声后的梯度
        """
        noisy_gradients = {}
        
        for key, grad in gradients.items():
            # 计算噪声标准差
            noise_std = self.noise_multiplier * self.l2_norm_clip
            
            # 生成高斯噪声
            noise = torch.randn_like(grad) * noise_std
            
            # 添加噪声
            noisy_gradients[key] = grad + noise
            
        return noisy_gradients
    
    def add_noise_to_model_updates(self, model_updates: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        向模型更新添加噪声
        
        Args:
            model_updates: 模型更新列表
            
        Returns:
            noisy_updates: 添加噪声后的更新
        """
        noisy_updates = []
        
        for update in model_updates:
            noisy_update = {}
            for key, tensor in update.items():
                # 计算噪声标准差
                noise_std = self.noise_multiplier * self.l2_norm_clip
                
                # 生成高斯噪声
                noise = torch.randn_like(tensor) * noise_std
                
                # 添加噪声
                noisy_update[key] = tensor + noise
                
            noisy_updates.append(noisy_update)
            
        return noisy_updates


class DPFedAvg:
    """
    带差分隐私的联邦平均算法
    """
    
    def __init__(self, model, lr=0.1, device='cpu', 
                 epsilon=1.0, delta=1e-5, noise_multiplier=None,
                 l2_norm_clip=1.0, sample_rate=1.0):
        self.model = model
        self.lr = lr
        self.device = device
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
        
        # 优化器和损失函数
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
    
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
    
    def get_model_state(self):
        """获取模型状态"""
        return self.model.state_dict()
    
    def set_model_state(self, state_dict):
        """设置模型状态"""
        self.model.load_state_dict(state_dict)
    
    def aggregate_updates(self, client_updates, client_weights=None):
        """
        聚合客户端更新（带噪声）
        
        Args:
            client_updates: 客户端更新列表
            client_weights: 客户端权重列表
        """
        if not client_updates:
            return
            
        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # 加权平均
        aggregated_state = {}
        for key in client_updates[0].keys():
            aggregated_state[key] = sum(
                client_updates[i][key] * client_weights[i] 
                for i in range(len(client_updates))
            )
        
        self.set_model_state(aggregated_state)
    
    def evaluate(self, test_dataset, batch_size=32):
        """评估模型性能"""
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        correct = 0
        total = 0
        total_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def get_privacy_spent(self):
        """获取已消耗的隐私预算"""
        return self.privacy_accountant.get_privacy_spent()
    
    def get_remaining_budget(self):
        """获取剩余隐私预算"""
        return self.privacy_accountant.get_remaining_budget()
    
    def is_budget_exhausted(self):
        """检查隐私预算是否耗尽"""
        return self.privacy_accountant.is_budget_exhausted()
