"""
完全匹配论文的PFA实现
实现论文中的所有细节，包括精确的客户端分类、Lanczos投影、异构DP等
"""

import torch
import copy
import numpy as np
from typing import List, Dict, Tuple, Optional
from .fedavg import FedAvg
from ..utils.client_division import PreciseClientDivision
from ..utils.lanczos_precise import PreciseLanczosProjection
from ..utils.aggregation_precise import PreciseAggregation
from ..privacy.heterogeneous_dp import HeterogeneousDP
from ..utils.lanczos import flatten_model_state, unflatten_model_state

class PFA_Precise(FedAvg):
    """
    完全匹配论文的PFA实现
    实现论文Algorithm 2和Algorithm 3的所有细节
    """
    
    def __init__(self, model, lr=0.1, proj_dims=1, lanczos_iter=256, device='cpu',
                 delay=False, clustering_method='kmeans', balance_ratio=0.1):
        """
        初始化精确的PFA算法
        
        Args:
            model: 模型
            lr: 学习率
            proj_dims: 投影维度
            lanczos_iter: Lanczos迭代次数
            device: 设备
            delay: 是否使用延迟投影
            clustering_method: 聚类方法
            balance_ratio: 公共客户端比例
        """
        super().__init__(model, lr, device)
        
        # PFA参数
        self.proj_dims = proj_dims
        self.lanczos_iter = lanczos_iter
        self.delay = delay
        
        # 客户端分类器
        self.client_division = PreciseClientDivision(
            clustering_method=clustering_method,
            balance_ratio=balance_ratio
        )
        
        # Lanczos投影器
        self.lanczos_projection = PreciseLanczosProjection(
            max_iter=lanczos_iter,
            tolerance=1e-12,
            reorthogonalize=True,
            check_convergence=True
        )
        
        # 聚合器
        self.aggregation = PreciseAggregation(
            aggregation_method='privacy_weighted',
            privacy_aware=True,
            dataset_size_aware=True
        )
        
        # 异构DP
        self.heterogeneous_dp = None
        
        # 状态管理
        self.global_model_state = None
        self.public_clients = []
        self.private_clients = []
        self.projection_matrix = None
        self.mean_vector = None
        
        # 更新历史
        self.public_updates = []
        self.private_updates = []
        self.num_public = 0
        self.num_private = 0
        
        # 投影质量监控
        self.projection_quality_history = []
    
    def set_heterogeneous_dp(self, client_epsilons: List[float], 
                           client_deltas: List[float]):
        """
        设置异构差分隐私参数
        """
        self.heterogeneous_dp = HeterogeneousDP(
            client_epsilons=client_epsilons,
            client_deltas=client_deltas
        )
    
    def divide_clients(self, epsilons: List[float], 
                      dataset_sizes: Optional[List[int]] = None,
                      additional_features: Optional[np.ndarray] = None):
        """
        客户端分类，完全匹配论文Algorithm 2步骤1
        """
        self.public_clients, self.private_clients = self.client_division.divide_clients(
            epsilons=epsilons,
            dataset_sizes=dataset_sizes,
            additional_features=additional_features
        )
        
        print(f"客户端分类完成:")
        print(f"  公共客户端: {self.public_clients}")
        print(f"  私有客户端: {self.private_clients}")
        
        return self.public_clients, self.private_clients
    
    def identify_subspace(self, public_updates: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        子空间识别，完全匹配论文Algorithm 3步骤2-3
        """
        if not public_updates:
            raise ValueError("公共客户端更新列表不能为空")
        
        # 将更新转换为向量
        update_vectors = []
        for update in public_updates:
            update_vector = flatten_model_state(update)
            update_vectors.append(update_vector)
        
        # 计算投影矩阵和均值向量
        projection_matrix, mean_vector = self.lanczos_projection.compute_projection_matrix(
            update_vectors, self.proj_dims
        )
        
        # 存储投影信息
        self.projection_matrix = projection_matrix
        self.mean_vector = mean_vector
        
        # 记录投影质量
        quality_metrics = self.lanczos_projection.estimate_projection_quality(
            update_vectors, [], projection_matrix, mean_vector
        )
        self.projection_quality_history.append(quality_metrics)
        
        print(f"子空间识别完成:")
        print(f"  投影矩阵形状: {projection_matrix.shape}")
        print(f"  投影质量: {quality_metrics}")
        
        return projection_matrix, mean_vector
    
    def project_private_updates(self, private_updates: List[Dict]) -> List[Dict]:
        """
        私有更新投影，完全匹配论文Algorithm 3步骤4
        """
        if not private_updates:
            return []
        
        if self.projection_matrix is None or self.mean_vector is None:
            raise ValueError("投影矩阵未初始化，请先调用identify_subspace")
        
        # 将更新转换为向量
        update_vectors = []
        for update in private_updates:
            update_vector = flatten_model_state(update)
            update_vectors.append(update_vector)
        
        # 投影到子空间
        projected_vectors = self.lanczos_projection.project_vectors(
            update_vectors, self.projection_matrix, self.mean_vector
        )
        
        # 重构到原始空间
        reconstructed_vectors = self.lanczos_projection.reconstruct_vectors(
            projected_vectors, self.projection_matrix, self.mean_vector
        )
        
        # 转换回模型状态字典
        projected_updates = []
        for i, reconstructed_vector in enumerate(reconstructed_vectors):
            projected_update = unflatten_model_state(
                reconstructed_vector, private_updates[i]
            )
            projected_updates.append(projected_update)
        
        print(f"私有更新投影完成: {len(projected_updates)}个更新")
        
        return projected_updates
    
    def aggregate_updates(self, client_updates: List[Dict], 
                         client_weights: Optional[List[float]] = None,
                         client_epsilons: Optional[List[float]] = None,
                         client_dataset_sizes: Optional[List[int]] = None,
                         client_types: Optional[List[str]] = None):
        """
        投影联邦平均，完全匹配论文Algorithm 3步骤5-6
        """
        if not client_updates:
            return
        
        # 保存当前全局模型状态
        self.global_model_state = copy.deepcopy(self.get_model_state())
        
        # 分离公共和私有更新
        public_updates = []
        private_updates = []
        public_weights = []
        private_weights = []
        public_epsilons = []
        private_epsilons = []
        public_dataset_sizes = []
        private_dataset_sizes = []
        
        for i, update in enumerate(client_updates):
            if i in self.public_clients:
                public_updates.append(update)
                if client_weights:
                    public_weights.append(client_weights[i])
                if client_epsilons:
                    public_epsilons.append(client_epsilons[i])
                if client_dataset_sizes:
                    public_dataset_sizes.append(client_dataset_sizes[i])
            else:
                private_updates.append(update)
                if client_weights:
                    private_weights.append(client_weights[i])
                if client_epsilons:
                    private_epsilons.append(client_epsilons[i])
                if client_dataset_sizes:
                    private_dataset_sizes.append(client_dataset_sizes[i])
        
        # 步骤1: 子空间识别（使用公共客户端更新）
        if public_updates:
            self.identify_subspace(public_updates)
        
        # 步骤2: 投影私有更新
        if private_updates and self.projection_matrix is not None:
            projected_private_updates = self.project_private_updates(private_updates)
        else:
            projected_private_updates = []
        
        # 步骤3: 计算聚合权重
        if not public_weights:
            public_weights = [1.0 / len(public_updates)] * len(public_updates) if public_updates else []
        if not private_weights:
            private_weights = [1.0 / len(projected_private_updates)] * len(projected_private_updates) if projected_private_updates else []
        
        # 步骤4: 加权聚合
        if public_updates and projected_private_updates:
            # 同时有公共和私有更新
            aggregated_update = self.aggregation.aggregate_with_projection(
                public_updates=public_updates,
                projected_private_updates=projected_private_updates,
                public_weights=public_weights,
                private_weights=private_weights
            )
        elif public_updates:
            # 只有公共更新
            aggregated_update = self.aggregation.aggregate_updates(
                public_updates, public_weights
            )
        elif projected_private_updates:
            # 只有私有更新
            aggregated_update = self.aggregation.aggregate_updates(
                projected_private_updates, private_weights
            )
        else:
            raise ValueError("没有可聚合的更新")
        
        # 步骤5: 应用更新到全局模型
        self._apply_aggregated_update(aggregated_update)
        
        print(f"投影联邦平均完成:")
        print(f"  公共更新: {len(public_updates)}")
        print(f"  私有更新: {len(projected_private_updates)}")
        print(f"  聚合信息: {self.aggregation.get_aggregation_info()}")
    
    def _apply_aggregated_update(self, aggregated_update: Dict[str, torch.Tensor]):
        """
        应用聚合后的更新到全局模型
        """
        new_global_state = {}
        for key in self.global_model_state.keys():
            if key in aggregated_update:
                new_global_state[key] = self.global_model_state[key] + aggregated_update[key]
            else:
                new_global_state[key] = self.global_model_state[key]
        
        self.set_model_state(new_global_state)
    
    def local_update_with_dp(self, dataset, local_steps=100, batch_size=4, 
                           client_id=0, l2_norm_clip=1.0):
        """
        带异构DP的本地更新
        """
        if self.heterogeneous_dp is None:
            # 没有DP，使用普通更新
            return self.local_update(dataset, local_steps, batch_size)
        
        # 检查客户端预算
        if not self.heterogeneous_dp.check_client_budget(client_id):
            print(f"客户端{client_id}隐私预算已耗尽")
            return self.get_model_state()
        
        # 计算噪声乘数
        dataset_size = len(dataset)
        noise_multipliers = self.heterogeneous_dp.compute_noise_multipliers(
            [dataset_size], [batch_size], [local_steps]
        )
        noise_multiplier = noise_multipliers[0]
        
        # 执行本地训练
        self.model.train()
        
        for step in range(local_steps):
            # 随机采样batch
            batch_indices = torch.randperm(dataset_size)[:batch_size]
            
            # 获取batch数据
            batch_data = []
            batch_targets = []
            for idx in batch_indices:
                data, target = dataset[idx]
                batch_data.append(data)
                batch_targets.append(target)
            
            data = torch.stack(batch_data).to(self.device)
            target = torch.stack(batch_targets).to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # 应用异构DP
            gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
            
            # 应用DP
            dp_gradients = self.heterogeneous_dp.apply_heterogeneous_dp(
                [gradients], [client_id], [noise_multiplier], l2_norm_clip
            )[0]
            
            # 更新参数
            for name, param in self.model.named_parameters():
                if name in dp_gradients:
                    param.grad = dp_gradients[name]
            
            self.optimizer.step()
        
        # 更新隐私预算
        self.heterogeneous_dp._update_client_budget(client_id, local_steps)
        
        return self.get_model_state()
    
    def get_privacy_guarantees(self) -> Dict[str, any]:
        """
        获取隐私保证信息
        """
        if self.heterogeneous_dp is None:
            return {}
        
        return self.heterogeneous_dp.get_heterogeneous_privacy_guarantee()
    
    def get_projection_quality(self) -> Dict[str, any]:
        """
        获取投影质量信息
        """
        if not self.projection_quality_history:
            return {}
        
        latest_quality = self.projection_quality_history[-1]
        convergence_info = self.lanczos_projection.get_convergence_info()
        
        return {
            'latest_quality': latest_quality,
            'convergence_info': convergence_info,
            'projection_matrix_shape': self.projection_matrix.shape if self.projection_matrix is not None else None,
            'mean_vector_shape': self.mean_vector.shape if self.mean_vector is not None else None
        }
    
    def get_algorithm_info(self) -> Dict[str, any]:
        """
        获取算法信息
        """
        return {
            'algorithm': 'PFA_Precise',
            'proj_dims': self.proj_dims,
            'lanczos_iter': self.lanczos_iter,
            'delay': self.delay,
            'public_clients': self.public_clients,
            'private_clients': self.private_clients,
            'client_division_info': self.client_division.get_classification_info(),
            'aggregation_info': self.aggregation.get_aggregation_info(),
            'privacy_guarantees': self.get_privacy_guarantees(),
            'projection_quality': self.get_projection_quality()
        }
    
    def reset_state(self):
        """
        重置算法状态
        """
        self.global_model_state = None
        self.public_clients = []
        self.private_clients = []
        self.projection_matrix = None
        self.mean_vector = None
        self.public_updates = []
        self.private_updates = []
        self.num_public = 0
        self.num_private = 0
        self.projection_quality_history = []
        
        # 重置组件状态
        self.client_division = PreciseClientDivision()
        self.aggregation.clear_history()
        
        if self.heterogeneous_dp is not None:
            self.heterogeneous_dp.reset_budgets()
