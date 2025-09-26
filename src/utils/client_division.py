"""
精确的客户端分类模块
完全匹配论文中的客户端分类逻辑，包括聚类分析和动态阈值调整
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class PreciseClientDivision:
    """
    精确的客户端分类器
    实现论文中的客户端分类逻辑，包括聚类分析和动态阈值调整
    """
    
    def __init__(self, clustering_method: str = 'kmeans', 
                 balance_ratio: float = 0.1,
                 min_public_clients: int = 1,
                 max_public_clients: Optional[int] = None):
        """
        初始化客户端分类器
        
        Args:
            clustering_method: 聚类方法 ('kmeans', 'gaussian_mixture')
            balance_ratio: 公共客户端比例
            min_public_clients: 最少公共客户端数量
            max_public_clients: 最多公共客户端数量
        """
        self.clustering_method = clustering_method
        self.balance_ratio = balance_ratio
        self.min_public_clients = min_public_clients
        self.max_public_clients = max_public_clients
        
        # 分类结果
        self.public_clients = []
        self.private_clients = []
        self.cluster_centers = None
        self.threshold = None
        
    def divide_clients(self, epsilons: List[float], 
                      dataset_sizes: Optional[List[int]] = None,
                      additional_features: Optional[np.ndarray] = None) -> Tuple[List[int], List[int]]:
        """
        根据隐私参数和数据集大小对客户端进行分类
        
        Args:
            epsilons: 客户端隐私参数列表
            dataset_sizes: 客户端数据集大小列表
            additional_features: 额外的特征矩阵 (n_clients, n_features)
            
        Returns:
            public_clients: 公共客户端索引列表
            private_clients: 私有客户端索引列表
        """
        epsilons = np.array(epsilons)
        n_clients = len(epsilons)
        
        # 准备特征矩阵
        features = self._prepare_features(epsilons, dataset_sizes, additional_features)
        
        # 执行聚类分析
        cluster_labels = self._perform_clustering(features)
        
        # 根据聚类结果和隐私参数确定公共/私有客户端
        public_clients, private_clients = self._assign_clients_to_groups(
            epsilons, cluster_labels, n_clients
        )
        
        # 验证分类结果
        public_clients, private_clients = self._validate_and_adjust_classification(
            public_clients, private_clients, epsilons, n_clients
        )
        
        self.public_clients = public_clients
        self.private_clients = private_clients
        
        print(f"客户端分类完成:")
        print(f"  公共客户端: {public_clients} (ε: {[epsilons[i] for i in public_clients]})")
        print(f"  私有客户端: {private_clients} (ε: {[epsilons[i] for i in private_clients]})")
        print(f"  分类阈值: {self.threshold:.4f}")
        
        return public_clients, private_clients
    
    def _prepare_features(self, epsilons: np.ndarray, 
                         dataset_sizes: Optional[List[int]], 
                         additional_features: Optional[np.ndarray]) -> np.ndarray:
        """
        准备聚类特征矩阵
        """
        features_list = [epsilons.reshape(-1, 1)]
        
        # 添加数据集大小特征
        if dataset_sizes is not None:
            dataset_sizes = np.array(dataset_sizes)
            # 标准化数据集大小
            dataset_sizes_norm = (dataset_sizes - np.mean(dataset_sizes)) / (np.std(dataset_sizes) + 1e-8)
            features_list.append(dataset_sizes_norm.reshape(-1, 1))
        
        # 添加额外特征
        if additional_features is not None:
            features_list.append(additional_features)
        
        # 合并所有特征
        features = np.hstack(features_list)
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        return features_scaled
    
    def _perform_clustering(self, features: np.ndarray) -> np.ndarray:
        """
        执行聚类分析
        """
        n_clients = features.shape[0]
        
        if self.clustering_method == 'kmeans':
            # 使用K-means聚类
            n_clusters = min(2, n_clients)  # 至少2个聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            self.cluster_centers = kmeans.cluster_centers_
            
        elif self.clustering_method == 'gaussian_mixture':
            # 使用高斯混合模型
            from sklearn.mixture import GaussianMixture
            n_components = min(2, n_clients)
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            cluster_labels = gmm.fit_predict(features)
            self.cluster_centers = gmm.means_
            
        else:
            raise ValueError(f"不支持的聚类方法: {self.clustering_method}")
        
        return cluster_labels
    
    def _assign_clients_to_groups(self, epsilons: np.ndarray, 
                                 cluster_labels: np.ndarray, 
                                 n_clients: int) -> Tuple[List[int], List[int]]:
        """
        根据聚类结果和隐私参数分配客户端到公共/私有组
        """
        # 计算每个聚类的平均隐私参数
        unique_clusters = np.unique(cluster_labels)
        cluster_epsilons = []
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_eps = epsilons[cluster_mask]
            cluster_epsilons.append(np.mean(cluster_eps))
        
        # 确定高隐私聚类（ε值较大的聚类）
        high_privacy_cluster = unique_clusters[np.argmax(cluster_epsilons)]
        
        # 初始分配
        public_candidates = np.where(cluster_labels == high_privacy_cluster)[0]
        private_candidates = np.where(cluster_labels != high_privacy_cluster)[0]
        
        # 如果高隐私聚类中的客户端太少，从其他聚类中补充
        target_public_count = max(self.min_public_clients, 
                                 int(self.balance_ratio * n_clients))
        
        if len(public_candidates) < target_public_count:
            # 从所有客户端中选择ε值最大的作为公共客户端
            sorted_indices = np.argsort(epsilons)[::-1]  # 降序排列
            public_candidates = sorted_indices[:target_public_count]
            private_candidates = sorted_indices[target_public_count:]
        
        # 限制公共客户端数量
        if self.max_public_clients is not None:
            if len(public_candidates) > self.max_public_clients:
                # 选择ε值最大的客户端
                public_eps = epsilons[public_candidates]
                top_indices = np.argsort(public_eps)[::-1][:self.max_public_clients]
                public_candidates = public_candidates[top_indices]
                private_candidates = np.setdiff1d(np.arange(n_clients), public_candidates)
        
        # 计算分类阈值
        if len(public_candidates) > 0:
            self.threshold = np.min(epsilons[public_candidates])
        else:
            self.threshold = np.median(epsilons)
        
        return public_candidates.tolist(), private_candidates.tolist()
    
    def _validate_and_adjust_classification(self, public_clients: List[int], 
                                          private_clients: List[int], 
                                          epsilons: np.ndarray, 
                                          n_clients: int) -> Tuple[List[int], List[int]]:
        """
        验证并调整分类结果
        """
        # 确保至少有一个公共客户端
        if len(public_clients) == 0:
            # 选择ε值最大的客户端作为公共客户端
            max_epsilon_idx = np.argmax(epsilons)
            public_clients = [max_epsilon_idx]
            private_clients = [i for i in range(n_clients) if i != max_epsilon_idx]
        
        # 确保至少有一个私有客户端
        if len(private_clients) == 0:
            # 选择ε值最小的客户端作为私有客户端
            min_epsilon_idx = np.argmin(epsilons)
            private_clients = [min_epsilon_idx]
            public_clients = [i for i in range(n_clients) if i != min_epsilon_idx]
        
        # 验证分类的合理性
        if len(public_clients) > 0 and len(private_clients) > 0:
            public_eps = epsilons[public_clients]
            private_eps = epsilons[private_clients]
            
            # 如果公共客户端的平均ε值小于私有客户端，需要调整
            if np.mean(public_eps) < np.mean(private_eps):
                print("警告: 公共客户端平均ε值小于私有客户端，正在调整...")
                # 重新分配：ε值大的作为公共客户端
                sorted_indices = np.argsort(epsilons)[::-1]
                mid_point = len(sorted_indices) // 2
                public_clients = sorted_indices[:mid_point].tolist()
                private_clients = sorted_indices[mid_point:].tolist()
        
        return public_clients, private_clients
    
    def get_classification_info(self) -> dict:
        """
        获取分类信息
        """
        return {
            'public_clients': self.public_clients,
            'private_clients': self.private_clients,
            'threshold': self.threshold,
            'cluster_centers': self.cluster_centers,
            'balance_ratio': self.balance_ratio
        }
    
    def update_classification(self, new_epsilons: List[float], 
                            dataset_sizes: Optional[List[int]] = None):
        """
        更新客户端分类（用于动态调整）
        """
        return self.divide_clients(new_epsilons, dataset_sizes)
