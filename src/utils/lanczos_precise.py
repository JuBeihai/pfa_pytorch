"""
精确的Lanczos投影模块
完全匹配论文中的Lanczos算法实现，包括数值稳定性保证和收敛性检查
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import warnings

class PreciseLanczosProjection:
    """
    精确的Lanczos投影实现
    完全匹配论文中的Lanczos算法，包括数值稳定性保证和收敛性检查
    """
    
    def __init__(self, max_iter: int = 256, tolerance: float = 1e-12, 
                 reorthogonalize: bool = True, check_convergence: bool = True):
        """
        初始化Lanczos投影器
        
        Args:
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            reorthogonalize: 是否重新正交化
            check_convergence: 是否检查收敛性
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.reorthogonalize = reorthogonalize
        self.check_convergence = check_convergence
        
        # 收敛信息
        self.convergence_info = {
            'iterations': 0,
            'converged': False,
            'residual_norm': float('inf'),
            'eigenvalue_accuracy': 0.0
        }
    
    def compute_projection_matrix(self, public_updates: List[torch.Tensor], 
                                proj_dims: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算投影矩阵，完全匹配论文的投影矩阵计算
        
        Args:
            public_updates: 公共客户端更新列表
            proj_dims: 投影维度
            
        Returns:
            projection_matrix: 投影矩阵 (n_features, proj_dims)
            mean_vector: 均值向量 (n_features,)
        """
        if not public_updates:
            raise ValueError("公共客户端更新列表不能为空")
        
        # 转换为numpy数组进行处理
        updates_np = [update.detach().cpu().numpy() if hasattr(update, 'detach') else update 
                     for update in public_updates]
        
        # 堆叠更新向量
        stacked_updates = np.stack(updates_np, axis=1)  # (n_features, n_clients)
        
        # 计算均值
        mean_vector = np.mean(stacked_updates, axis=1)
        
        # 中心化更新
        centered_updates = stacked_updates - mean_vector.reshape(-1, 1)
        
        # 计算协方差矩阵
        n_clients = len(public_updates)
        if n_clients > 1:
            covariance_matrix = np.dot(centered_updates, centered_updates.T) / (n_clients - 1)
        else:
            # 单客户端情况，使用外积
            covariance_matrix = np.outer(centered_updates[:, 0], centered_updates[:, 0])
        
        # 使用Lanczos算法计算特征向量
        projection_matrix = self._lanczos_algorithm(covariance_matrix, proj_dims)
        
        # 转换回torch张量
        projection_matrix = torch.from_numpy(projection_matrix).float()
        mean_vector = torch.from_numpy(mean_vector).float()
        
        return projection_matrix, mean_vector
    
    def _lanczos_algorithm(self, A: np.ndarray, k: int) -> np.ndarray:
        """
        真正的Lanczos算法实现
        
        Args:
            A: 对称矩阵 (n x n)
            k: 需要的特征向量数量
            
        Returns:
            V: 特征向量矩阵 (n x k)
        """
        n = A.shape[0]
        k = min(k, n, self.max_iter)
        
        # 初始化
        V = np.zeros((n, k))
        alpha = np.zeros(k)
        beta = np.zeros(k-1)
        
        # 随机初始化向量
        np.random.seed(42)  # 保证可重复性
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        V[:, 0] = v
        
        # Lanczos迭代
        for i in range(k):
            # 计算 Av
            Av = np.dot(A, v)
            
            # 计算 alpha[i] = v^T * A * v
            alpha[i] = np.dot(v, Av)
            
            if i < k - 1:
                # 计算 w = Av - alpha[i]*v - beta[i-1]*v_prev
                w = Av - alpha[i] * v
                if i > 0:
                    w = w - beta[i-1] * V[:, i-1]
                
                # 重新正交化（提高数值稳定性）
                if self.reorthogonalize and i > 0:
                    w = self._reorthogonalize(w, V[:, :i+1])
                
                # 计算 beta[i] = ||w||
                beta_norm = np.linalg.norm(w)
                beta[i] = beta_norm
                
                # 检查收敛
                if self.check_convergence and beta_norm < self.tolerance:
                    print(f"Lanczos算法在第{i+1}步收敛")
                    self.convergence_info['converged'] = True
                    self.convergence_info['iterations'] = i + 1
                    self.convergence_info['residual_norm'] = beta_norm
                    break
                
                # 更新 v
                if beta_norm > self.tolerance:
                    v = w / beta_norm
                    V[:, i+1] = v
                else:
                    # 如果w接近零向量，使用随机向量
                    v = np.random.randn(n)
                    v = v / np.linalg.norm(v)
                    V[:, i+1] = v
        
        # 构建三对角矩阵T
        T = self._build_tridiagonal_matrix(alpha, beta, k)
        
        # 计算T的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(T)
        
        # 选择最大的特征值对应的特征向量
        sorted_indices = np.argsort(eigenvalues)[::-1]
        selected_indices = sorted_indices[:min(proj_dims, len(eigenvalues))]
        
        # 计算原始空间的特征向量
        V_selected = V[:, :len(selected_indices)]
        U_selected = eigenvectors[:, selected_indices]
        
        # 原始空间的特征向量 = V * U
        projection_matrix = np.dot(V_selected, U_selected)
        
        # 更新收敛信息
        self.convergence_info['iterations'] = k
        self.convergence_info['eigenvalue_accuracy'] = self._estimate_eigenvalue_accuracy(
            A, projection_matrix, eigenvalues[selected_indices]
        )
        
        return projection_matrix
    
    def _reorthogonalize(self, w: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        重新正交化向量w相对于V的列
        """
        for i in range(V.shape[1]):
            w = w - np.dot(w, V[:, i]) * V[:, i]
        return w
    
    def _build_tridiagonal_matrix(self, alpha: np.ndarray, beta: np.ndarray, k: int) -> np.ndarray:
        """
        构建三对角矩阵T
        """
        T = np.diag(alpha[:k])
        for i in range(min(k-1, len(beta))):
            T[i, i+1] = beta[i]
            T[i+1, i] = beta[i]
        return T
    
    def _estimate_eigenvalue_accuracy(self, A: np.ndarray, V: np.ndarray, 
                                    eigenvalues: np.ndarray) -> float:
        """
        估计特征值计算的准确性
        """
        if V.shape[1] == 0:
            return 0.0
        
        # 计算残差范数
        AV = np.dot(A, V)
        V_eigenvals = V * eigenvalues.reshape(1, -1)
        residual = AV - V_eigenvals
        
        # 计算相对残差
        residual_norm = np.linalg.norm(residual, 'fro')
        AV_norm = np.linalg.norm(AV, 'fro')
        
        if AV_norm > 0:
            relative_residual = residual_norm / AV_norm
        else:
            relative_residual = 0.0
        
        return 1.0 - relative_residual
    
    def project_vectors(self, vectors: List[torch.Tensor], 
                       projection_matrix: torch.Tensor, 
                       mean_vector: torch.Tensor) -> List[torch.Tensor]:
        """
        将向量投影到子空间
        
        Args:
            vectors: 要投影的向量列表
            projection_matrix: 投影矩阵
            mean_vector: 均值向量
            
        Returns:
            projected_vectors: 投影后的向量列表
        """
        projected_vectors = []
        
        for vector in vectors:
            # 转换为numpy数组
            if hasattr(vector, 'detach'):
                vec_np = vector.detach().cpu().numpy()
            else:
                vec_np = vector
            
            # 中心化
            centered_vec = vec_np - mean_vector.numpy()
            
            # 投影
            projected_vec = np.dot(projection_matrix.T, centered_vec)
            
            # 转换回torch张量
            projected_vec = torch.from_numpy(projected_vec).float()
            projected_vectors.append(projected_vec)
        
        return projected_vectors
    
    def reconstruct_vectors(self, projected_vectors: List[torch.Tensor], 
                           projection_matrix: torch.Tensor, 
                           mean_vector: torch.Tensor) -> List[torch.Tensor]:
        """
        从投影重构原始向量
        
        Args:
            projected_vectors: 投影后的向量列表
            projection_matrix: 投影矩阵
            mean_vector: 均值向量
            
        Returns:
            reconstructed_vectors: 重构的向量列表
        """
        reconstructed_vectors = []
        
        for proj_vec in projected_vectors:
            # 转换为numpy数组
            if hasattr(proj_vec, 'detach'):
                proj_np = proj_vec.detach().cpu().numpy()
            else:
                proj_np = proj_vec
            
            # 重构
            reconstructed_vec = np.dot(projection_matrix, proj_np) + mean_vector.numpy()
            
            # 转换回torch张量
            reconstructed_vec = torch.from_numpy(reconstructed_vec).float()
            reconstructed_vectors.append(reconstructed_vec)
        
        return reconstructed_vectors
    
    def get_convergence_info(self) -> dict:
        """
        获取收敛信息
        """
        return self.convergence_info.copy()
    
    def estimate_projection_quality(self, original_vectors: List[torch.Tensor], 
                                  projected_vectors: List[torch.Tensor], 
                                  projection_matrix: torch.Tensor, 
                                  mean_vector: torch.Tensor) -> dict:
        """
        估计投影质量
        
        Returns:
            quality_metrics: 投影质量指标
        """
        # 重构原始向量
        reconstructed_vectors = self.reconstruct_vectors(
            projected_vectors, projection_matrix, mean_vector
        )
        
        # 计算重构误差
        reconstruction_errors = []
        for orig, recon in zip(original_vectors, reconstructed_vectors):
            error = torch.norm(orig - recon).item()
            reconstruction_errors.append(error)
        
        # 计算投影保持的信息量
        original_norm = sum(torch.norm(v).item() for v in original_vectors)
        projected_norm = sum(torch.norm(v).item() for v in projected_vectors)
        information_ratio = projected_norm / original_norm if original_norm > 0 else 0.0
        
        return {
            'mean_reconstruction_error': np.mean(reconstruction_errors),
            'max_reconstruction_error': np.max(reconstruction_errors),
            'information_ratio': information_ratio,
            'compression_ratio': len(projected_vectors[0]) / len(original_vectors[0]) if original_vectors else 0.0
        }
