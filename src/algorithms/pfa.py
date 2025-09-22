import torch
import copy
from typing import List, Dict, Tuple
from .fedavg import FedAvg
from ..utils.lanczos import LanczosProjection, flatten_model_state, unflatten_model_state

class PFA(FedAvg):
    """
    Projected Federated Averaging (PFA) 算法
    使用 Lanczos 投影减少通信开销
    """
    
    def __init__(self, model, lr=0.1, proj_dims=1, lanczos_iter=64, device='cpu'):
        super().__init__(model, lr, device)
        self.proj_dims = proj_dims
        self.lanczos_iter = lanczos_iter
        self.lanczos_projection = LanczosProjection(max_iter=lanczos_iter)
        self.global_model_state = None
        
    def compute_projection(self, client_updates: List[Dict]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        计算客户端更新的投影
        
        Args:
            client_updates: 客户端模型更新列表
            
        Returns:
            projection_matrix: 投影矩阵
            projected_updates: 投影后的更新
        """
        if not client_updates:
            raise ValueError("客户端更新列表不能为空")
        
        # 计算每个客户端的更新向量（相对于全局模型）
        update_vectors = []
        for update in client_updates:
            if self.global_model_state is None:
                # 如果没有全局模型状态，直接使用更新
                update_vector = flatten_model_state(update)
            else:
                # 计算相对于全局模型的更新
                update_vector = self._compute_update_vector(update)
            update_vectors.append(update_vector)
        
        # 使用 Lanczos 投影
        projection_matrix, projected_updates = self.lanczos_projection.project_to_subspace(
            update_vectors, self.proj_dims
        )
        
        return projection_matrix, projected_updates
    
    def _compute_update_vector(self, client_model_state: Dict) -> torch.Tensor:
        """
        计算客户端模型相对于全局模型的更新向量
        
        Args:
            client_model_state: 客户端模型状态
            
        Returns:
            update_vector: 更新向量
        """
        update_dict = {}
        for key in client_model_state.keys():
            if key in self.global_model_state:
                update_dict[key] = client_model_state[key] - self.global_model_state[key]
            else:
                update_dict[key] = client_model_state[key]
        
        return flatten_model_state(update_dict)
    
    def aggregate_updates(self, client_updates: List[Dict], client_weights: List[float] = None):
        """
        PFA 聚合：先投影，再聚合，最后重构
        
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
        
        # 重构到原始空间
        aggregated_update_vector = projection_matrix @ aggregated_projected
        
        # 将更新向量转换回模型状态字典
        aggregated_update_dict = unflatten_model_state(
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
    
    def get_compression_ratio(self) -> float:
        """
        计算压缩比
        
        Returns:
            compression_ratio: 压缩比 (原始维度 / 投影维度)
        """
        if self.global_model_state is None:
            return 1.0
        
        total_params = sum(p.numel() for p in self.global_model_state.values())
        return total_params / self.proj_dims
    
    def set_projection_dims(self, proj_dims: int):
        """
        设置投影维度
        
        Args:
            proj_dims: 投影维度
        """
        self.proj_dims = proj_dims
        self.lanczos_projection = LanczosProjection(max_iter=self.lanczos_iter)