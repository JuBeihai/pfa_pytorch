"""
精确的聚合权重计算模块
完全匹配论文中的加权聚合实现，考虑客户端数据集大小和隐私参数
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

class PreciseAggregation:
    """
    精确的聚合实现
    完全匹配论文中的加权聚合，考虑客户端数据集大小和隐私参数
    """
    
    def __init__(self, aggregation_method: str = 'weighted', 
                 privacy_aware: bool = True,
                 dataset_size_aware: bool = True):
        """
        初始化聚合器
        
        Args:
            aggregation_method: 聚合方法 ('weighted', 'uniform', 'privacy_weighted')
            privacy_aware: 是否考虑隐私参数
            dataset_size_aware: 是否考虑数据集大小
        """
        self.aggregation_method = aggregation_method
        self.privacy_aware = privacy_aware
        self.dataset_size_aware = dataset_size_aware
        
        # 聚合历史
        self.aggregation_history = []
    
    def compute_client_weights(self, client_updates: List[Dict], 
                             client_epsilons: Optional[List[float]] = None,
                             client_dataset_sizes: Optional[List[int]] = None,
                             client_types: Optional[List[str]] = None) -> List[float]:
        """
        计算客户端权重，完全匹配论文的权重计算
        
        Args:
            client_updates: 客户端更新列表
            client_epsilons: 客户端隐私参数列表
            client_dataset_sizes: 客户端数据集大小列表
            client_types: 客户端类型列表 ('public', 'private')
            
        Returns:
            client_weights: 客户端权重列表
        """
        n_clients = len(client_updates)
        
        if self.aggregation_method == 'uniform':
            # 均匀权重
            weights = [1.0 / n_clients] * n_clients
            
        elif self.aggregation_method == 'weighted':
            # 基于数据集大小的权重
            weights = self._compute_dataset_size_weights(client_dataset_sizes, n_clients)
            
        elif self.aggregation_method == 'privacy_weighted':
            # 基于隐私参数的权重
            weights = self._compute_privacy_aware_weights(
                client_epsilons, client_dataset_sizes, client_types, n_clients
            )
            
        else:
            raise ValueError(f"不支持的聚合方法: {self.aggregation_method}")
        
        # 归一化权重
        weights = self._normalize_weights(weights)
        
        # 记录聚合信息
        self.aggregation_history.append({
            'method': self.aggregation_method,
            'weights': weights.copy(),
            'n_clients': n_clients,
            'privacy_aware': self.privacy_aware,
            'dataset_size_aware': self.dataset_size_aware
        })
        
        return weights
    
    def _compute_dataset_size_weights(self, dataset_sizes: Optional[List[int]], 
                                    n_clients: int) -> List[float]:
        """
        基于数据集大小计算权重
        """
        if not self.dataset_size_aware or dataset_sizes is None:
            return [1.0 / n_clients] * n_clients
        
        # 使用数据集大小作为权重
        weights = [float(size) for size in dataset_sizes]
        
        # 处理零权重情况
        if all(w == 0 for w in weights):
            weights = [1.0 / n_clients] * n_clients
        
        return weights
    
    def _compute_privacy_aware_weights(self, epsilons: Optional[List[float]], 
                                     dataset_sizes: Optional[List[int]], 
                                     client_types: Optional[List[str]], 
                                     n_clients: int) -> List[float]:
        """
        基于隐私参数计算权重
        """
        if not self.privacy_aware or epsilons is None:
            return self._compute_dataset_size_weights(dataset_sizes, n_clients)
        
        # 基础权重（基于数据集大小）
        base_weights = self._compute_dataset_size_weights(dataset_sizes, n_clients)
        
        # 隐私调整因子
        privacy_factors = self._compute_privacy_factors(epsilons, client_types)
        
        # 组合权重
        weights = [base * privacy for base, privacy in zip(base_weights, privacy_factors)]
        
        return weights
    
    def _compute_privacy_factors(self, epsilons: List[float], 
                               client_types: Optional[List[str]]) -> List[float]:
        """
        计算隐私调整因子
        """
        epsilons = np.array(epsilons)
        
        if client_types is None:
            # 没有客户端类型信息，基于ε值调整
            # ε值越大，权重越大（隐私保护越少，贡献越大）
            max_epsilon = np.max(epsilons)
            min_epsilon = np.min(epsilons)
            
            if max_epsilon > min_epsilon:
                # 归一化到[0.5, 1.5]范围
                normalized_eps = (epsilons - min_epsilon) / (max_epsilon - min_epsilon)
                privacy_factors = 0.5 + normalized_eps
            else:
                privacy_factors = np.ones_like(epsilons)
        else:
            # 有客户端类型信息
            privacy_factors = []
            for i, client_type in enumerate(client_types):
                if client_type == 'public':
                    # 公共客户端：基于ε值，但权重较高
                    epsilon = epsilons[i]
                    max_epsilon = np.max(epsilons)
                    if max_epsilon > 0:
                        factor = 1.0 + (epsilon / max_epsilon) * 0.5  # [1.0, 1.5]
                    else:
                        factor = 1.0
                else:  # private
                    # 私有客户端：基于ε值，但权重较低
                    epsilon = epsilons[i]
                    max_epsilon = np.max(epsilons)
                    if max_epsilon > 0:
                        factor = 0.5 + (epsilon / max_epsilon) * 0.5  # [0.5, 1.0]
                    else:
                        factor = 0.5
                privacy_factors.append(factor)
        
        return privacy_factors
    
    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """
        归一化权重
        """
        total_weight = sum(weights)
        if total_weight > 0:
            return [w / total_weight for w in weights]
        else:
            n = len(weights)
            return [1.0 / n] * n
    
    def aggregate_updates(self, client_updates: List[Dict], 
                         client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        聚合客户端更新
        
        Args:
            client_updates: 客户端更新列表
            client_weights: 客户端权重列表
            
        Returns:
            aggregated_update: 聚合后的更新
        """
        if not client_updates:
            raise ValueError("客户端更新列表不能为空")
        
        if len(client_updates) != len(client_weights):
            raise ValueError("客户端更新数量与权重数量不匹配")
        
        # 获取所有参数键
        all_keys = set()
        for update in client_updates:
            all_keys.update(update.keys())
        
        # 聚合每个参数
        aggregated_update = {}
        for key in all_keys:
            # 收集该参数的所有更新
            param_updates = []
            param_weights = []
            
            for i, update in enumerate(client_updates):
                if key in update:
                    param_updates.append(update[key])
                    param_weights.append(client_weights[i])
            
            if param_updates:
                # 加权平均
                aggregated_param = self._weighted_average(param_updates, param_weights)
                aggregated_update[key] = aggregated_param
        
        return aggregated_update
    
    def _weighted_average(self, tensors: List[torch.Tensor], 
                         weights: List[float]) -> torch.Tensor:
        """
        计算张量的加权平均
        """
        if not tensors:
            raise ValueError("张量列表不能为空")
        
        # 确保所有权重为正
        weights = [max(w, 1e-8) for w in weights]
        
        # 归一化权重
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # 加权平均
        result = torch.zeros_like(tensors[0])
        for tensor, weight in zip(tensors, normalized_weights):
            result += weight * tensor
        
        return result
    
    def aggregate_with_projection(self, public_updates: List[Dict], 
                                projected_private_updates: List[Dict],
                                public_weights: List[float],
                                private_weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        聚合公共和投影后的私有更新
        
        Args:
            public_updates: 公共客户端更新
            projected_private_updates: 投影后的私有客户端更新
            public_weights: 公共客户端权重
            private_weights: 私有客户端权重
            
        Returns:
            aggregated_update: 聚合后的更新
        """
        # 分别聚合公共和私有更新
        if public_updates:
            aggregated_public = self.aggregate_updates(public_updates, public_weights)
        else:
            aggregated_public = {}
        
        if projected_private_updates:
            aggregated_private = self.aggregate_updates(projected_private_updates, private_weights)
        else:
            aggregated_private = {}
        
        # 合并公共和私有更新
        all_keys = set(aggregated_public.keys()) | set(aggregated_private.keys())
        final_update = {}
        
        for key in all_keys:
            if key in aggregated_public and key in aggregated_private:
                # 两个组都有该参数，进行加权平均
                total_public_weight = sum(public_weights)
                total_private_weight = sum(private_weights)
                total_weight = total_public_weight + total_private_weight
                
                if total_weight > 0:
                    public_weight = total_public_weight / total_weight
                    private_weight = total_private_weight / total_weight
                    
                    final_update[key] = (public_weight * aggregated_public[key] + 
                                       private_weight * aggregated_private[key])
                else:
                    final_update[key] = aggregated_public[key]
            elif key in aggregated_public:
                final_update[key] = aggregated_public[key]
            else:
                final_update[key] = aggregated_private[key]
        
        return final_update
    
    def get_aggregation_info(self) -> dict:
        """
        获取聚合信息
        """
        if not self.aggregation_history:
            return {}
        
        latest = self.aggregation_history[-1]
        return {
            'method': latest['method'],
            'n_clients': latest['n_clients'],
            'weights': latest['weights'],
            'weight_stats': {
                'min': min(latest['weights']),
                'max': max(latest['weights']),
                'mean': np.mean(latest['weights']),
                'std': np.std(latest['weights'])
            },
            'privacy_aware': latest['privacy_aware'],
            'dataset_size_aware': latest['dataset_size_aware']
        }
    
    def clear_history(self):
        """
        清空聚合历史
        """
        self.aggregation_history.clear()
