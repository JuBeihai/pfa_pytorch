import torch
import numpy as np
from typing import List, Tuple, Optional

class LanczosProjection:
    """
    Lanczos 算法实现，用于 PFA 中的低维投影
    """
    
    def __init__(self, max_iter: int = 64, tolerance: float = 1e-6):
        self.max_iter = max_iter
        self.tolerance = tolerance
    
    def lanczos_iteration(self, A: torch.Tensor, v0: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lanczos 迭代算法
        
        Args:
            A: 对称矩阵 (n x n)
            v0: 初始向量 (n,)
            k: 迭代次数
            
        Returns:
            V: 正交向量矩阵 (n x k)
            T: 三对角矩阵 (k x k)
        """
        n = A.shape[0]
        device = A.device
        
        # 初始化
        V = torch.zeros(n, k, device=device)
        alpha = torch.zeros(k, device=device)
        beta = torch.zeros(k-1, device=device)
        
        # 归一化初始向量
        v = v0 / torch.norm(v0)
        V[:, 0] = v
        
        for i in range(k):
            # 计算 Av
            Av = torch.mv(A, v)
            
            # 计算 alpha[i] = v^T * A * v
            alpha[i] = torch.dot(v, Av)
            
            if i < k - 1:
                # 计算 w = Av - alpha[i] * v - beta[i-1] * v_prev
                w = Av - alpha[i] * v
                if i > 0:
                    w = w - beta[i-1] * V[:, i-1]
                
                # 计算 beta[i] = ||w||
                beta[i] = torch.norm(w)
                
                # 检查收敛
                if beta[i] < self.tolerance:
                    break
                
                # 更新 v
                v = w / beta[i]
                V[:, i+1] = v
        
        # 构建三对角矩阵 T
        T = torch.diag(alpha)
        for i in range(k-1):
            T[i, i+1] = beta[i]
            T[i+1, i] = beta[i]
        
        return V, T
    
    def compute_eigenvalues(self, T: torch.Tensor) -> torch.Tensor:
        """
        计算三对角矩阵的特征值
        
        Args:
            T: 三对角矩阵 (k x k)
            
        Returns:
            eigenvalues: 特征值 (k,)
        """
        # 使用 PyTorch 的 eig 函数
        eigenvalues, _ = torch.linalg.eig(T)
        return eigenvalues.real
    
    def project_to_subspace(self, vectors: List[torch.Tensor], proj_dims: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将向量列表投影到低维子空间
        使用内存友好的方法，避免构建完整的 Gram 矩阵
        
        Args:
            vectors: 向量列表，每个向量形状为 (n,)
            proj_dims: 投影维度
            
        Returns:
            projection_matrix: 投影矩阵 (n x proj_dims)
            projected_vectors: 投影后的向量列表
        """
        if not vectors:
            raise ValueError("向量列表不能为空")
        
        device = vectors[0].device
        n = vectors[0].shape[0]
        
        # 对于大模型，使用简化的投影方法
        if n > 100000:  # 如果参数数量超过 100k，使用简化方法
            return self._simplified_projection(vectors, proj_dims)
        
        # 构建 Gram 矩阵 G = sum(v_i * v_i^T)
        G = torch.zeros(n, n, device=device)
        for v in vectors:
            G += torch.outer(v, v)
        
        # 使用随机初始向量
        v0 = torch.randn(n, device=device)
        v0 = v0 / torch.norm(v0)
        
        # Lanczos 迭代
        V, T = self.lanczos_iteration(G, v0, min(proj_dims, self.max_iter))
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = torch.linalg.eig(T)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        
        # 选择最大的特征值对应的特征向量
        _, indices = torch.topk(eigenvalues, min(proj_dims, len(eigenvalues)))
        projection_matrix = V @ eigenvectors[:, indices]
        
        # 投影向量
        projected_vectors = []
        for v in vectors:
            projected_v = projection_matrix.T @ v
            projected_vectors.append(projected_v)
        
        return projection_matrix, projected_vectors
    
    def _simplified_projection(self, vectors: List[torch.Tensor], proj_dims: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        简化的投影方法，用于大模型
        使用随机投影而不是 Lanczos 算法
        
        Args:
            vectors: 向量列表
            proj_dims: 投影维度
            
        Returns:
            projection_matrix: 投影矩阵
            projected_vectors: 投影后的向量列表
        """
        device = vectors[0].device
        n = vectors[0].shape[0]
        
        # 使用随机投影矩阵
        torch.manual_seed(42)  # 固定随机种子以保证可重复性
        projection_matrix = torch.randn(n, proj_dims, device=device)
        
        # 正交化投影矩阵
        projection_matrix, _ = torch.linalg.qr(projection_matrix, mode='reduced')
        
        # 投影向量
        projected_vectors = []
        for v in vectors:
            projected_v = projection_matrix.T @ v
            projected_vectors.append(projected_v)
        
        return projection_matrix, projected_vectors
    
    def reconstruct_from_projection(self, projection_matrix: torch.Tensor, 
                                  projected_vectors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        从投影重构原始向量
        
        Args:
            projection_matrix: 投影矩阵 (n x proj_dims)
            projected_vectors: 投影后的向量列表
            
        Returns:
            reconstructed_vectors: 重构的向量列表
        """
        reconstructed_vectors = []
        for proj_v in projected_vectors:
            recon_v = projection_matrix @ proj_v
            reconstructed_vectors.append(recon_v)
        
        return reconstructed_vectors


def compute_model_difference(model1_state: dict, model2_state: dict) -> torch.Tensor:
    """
    计算两个模型状态之间的差异
    
    Args:
        model1_state: 第一个模型的状态字典
        model2_state: 第二个模型的状态字典
        
    Returns:
        difference: 模型差异向量
    """
    differences = []
    
    for key in model1_state.keys():
        if key in model2_state:
            diff = model1_state[key] - model2_state[key]
            differences.append(diff.flatten())
    
    return torch.cat(differences)


def flatten_model_state(model_state: dict) -> torch.Tensor:
    """
    将模型状态字典展平为向量
    
    Args:
        model_state: 模型状态字典
        
    Returns:
        flattened: 展平的向量
    """
    tensors = []
    for key, value in model_state.items():
        tensors.append(value.flatten())
    
    return torch.cat(tensors)


def unflatten_model_state(flattened: torch.Tensor, model_state_template: dict) -> dict:
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
