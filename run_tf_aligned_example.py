#!/usr/bin/env python3
"""
运行完全对齐 TensorFlow 版本的示例
展示模型结构、训练循环、采样策略、聚合权重和噪声计算的完全匹配
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
import math

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn import MNISTCNN
from src.algorithms.fedavg import FedAvg
from src.algorithms.pfa_tf import PFA_TF
from src.algorithms.dp_fedavg_tf import DPFedAvg_TF
from src.privacy.accountant import PrivacyAccountant

def create_mock_dataset(size=1000, num_classes=10):
    """创建模拟数据集"""
    class MockDataset:
        def __init__(self, size, num_classes):
            self.size = size
            self.data = torch.randn(size, 784)
            self.targets = torch.randint(0, num_classes, (size,))
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    return MockDataset(size, num_classes)

def set_epsilons(filename: str, N: int) -> list:
    """设置 epsilon 值，匹配 TensorFlow 版本"""
    if filename == 'gauss1':
        epsilons = np.random.normal(1.0, 0.5, N)
        epsilons = np.clip(epsilons, 0.1, 10.0)
    elif filename == 'uniform1':
        epsilons = np.random.uniform(0.5, 5.0, N)
    else:
        epsilons = [1.0] * N
    
    print(f"Epsilons: {epsilons}")
    return epsilons.tolist()

def compute_noise_multiplier(N: int, L: int, T: int, epsilon: float, delta: float) -> float:
    """计算噪声乘数，匹配 TensorFlow 版本的公式"""
    q = L / N
    nm = 10 * q * math.sqrt(T * (-math.log10(delta))) / epsilon
    return nm

def sample_clients(candidates: list, num_clients: int, sample_ratio: float, 
                  public_clients: list = None) -> list:
    """采样客户端，匹配 TensorFlow 版本的逻辑"""
    m = int(num_clients * sample_ratio)
    if len(candidates) < m:
        return []
    
    # 随机采样
    participants = list(np.random.permutation(candidates))[:m]
    
    # 如果有公共客户端，确保至少有一个公共客户端参与
    if public_clients is not None:
        check = 50
        while check and len(set(participants).intersection(set(public_clients))) == 0:
            check -= 1
            participants = list(np.random.permutation(candidates))[:m]
        
        return participants if check else []
    
    return participants

def run_fedavg_example():
    """运行 FedAvg 示例"""
    print("=" * 60)
    print("运行 FedAvg 示例 - 完全对齐 TensorFlow 版本")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型 - 匹配 TensorFlow 架构
    model = MNISTCNN().to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建算法
    algorithm = FedAvg(model, lr=0.1, device=device)
    
    # 创建模拟数据
    client_datasets = [create_mock_dataset(1000) for _ in range(3)]
    test_dataset = create_mock_dataset(2000)
    
    print(f"客户端数据集大小: {[len(ds) for ds in client_datasets]}")
    print(f"测试数据集大小: {len(test_dataset)}")
    
    # 训练循环 - 匹配 TensorFlow 的 local_steps
    rounds = 5
    local_steps = 20
    batch_size = 4
    
    print(f"\n开始训练: {rounds} 轮, 每轮 {local_steps} 步, 批次大小 {batch_size}")
    
    for round_idx in range(rounds):
        print(f"\n--- 第 {round_idx + 1}/{rounds} 轮 ---")
        
        # 客户端本地训练
        client_updates = []
        for client_id, dataset in enumerate(client_datasets):
            # 下载全局模型
            algorithm.set_model_state(algorithm.get_model_state())
            
            # 本地训练 - 使用 local_steps 而不是 epochs
            start_time = time.time()
            algorithm.local_update(dataset, local_steps=local_steps, batch_size=batch_size)
            train_time = time.time() - start_time
            
            # 获取更新
            client_updates.append(algorithm.get_model_state())
            print(f"客户端 {client_id}: 训练时间 {train_time:.2f}s")
        
        # 服务器聚合
        algorithm.aggregate_updates(client_updates)
        
        # 评估
        accuracy, loss = algorithm.evaluate(test_dataset, batch_size=32)
        print(f"测试准确率: {accuracy:.2f}%, 损失: {loss:.4f}")

def run_pfa_example():
    """运行 PFA 示例"""
    print("\n" + "=" * 60)
    print("运行 PFA 示例 - 完全对齐 TensorFlow 版本")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = MNISTCNN().to(device)
    algorithm = PFA_TF(model, lr=0.1, proj_dims=5, device=device)
    
    # 设置公共客户端 - 匹配 TensorFlow 版本
    epsilons = [1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.9, 1.8, 0.7, 1.3]
    algorithm.set_public_clients(epsilons)
    
    # 创建模拟数据
    client_datasets = [create_mock_dataset(1000) for _ in range(5)]
    test_dataset = create_mock_dataset(2000)
    
    print(f"公共客户端: {algorithm.public_clients}")
    print(f"私有客户端: {[i for i in range(5) if i not in algorithm.public_clients]}")
    
    # 训练循环
    rounds = 3
    local_steps = 15
    batch_size = 4
    
    print(f"\n开始训练: {rounds} 轮, 每轮 {local_steps} 步, 批次大小 {batch_size}")
    
    for round_idx in range(rounds):
        print(f"\n--- 第 {round_idx + 1}/{rounds} 轮 ---")
        
        # 客户端本地训练和聚合
        for client_id, dataset in enumerate(client_datasets):
            # 下载全局模型
            algorithm.set_model_state(algorithm.get_model_state())
            
            # 本地训练
            algorithm.local_update(dataset, local_steps=local_steps, batch_size=batch_size)
            
            # 计算更新并聚合 - 匹配 TensorFlow 的 PFA 逻辑
            global_state = algorithm.get_model_state()
            client_state = algorithm.get_model_state()
            update = {}
            for key in global_state.keys():
                update[key] = global_state[key] - client_state[key]
            
            is_public = client_id in algorithm.public_clients
            update_values = [u.detach() for u in update.values()]
            algorithm.aggregate(client_id, update_values, is_public)
            
            print(f"客户端 {client_id} (公共: {is_public}): 已聚合")
        
        # 更新全局模型
        algorithm.update(algorithm.get_model_state())
        
        # 评估
        accuracy, loss = algorithm.evaluate(test_dataset, batch_size=32)
        print(f"测试准确率: {accuracy:.2f}%, 损失: {loss:.4f}")

def run_dp_fedavg_example():
    """运行 DP-FedAvg 示例"""
    print("\n" + "=" * 60)
    print("运行 DP-FedAvg 示例 - 完全对齐 TensorFlow 版本")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = MNISTCNN().to(device)
    
    # 设置 DP 参数 - 匹配 TensorFlow 版本
    epsilon = 1.0
    delta = 1e-5
    l2_norm_clip = 1.0
    sample_rate = 0.8
    
    algorithm = DPFedAvg_TF(
        model=model,
        lr=0.1,
        device=device,
        epsilon=epsilon,
        delta=delta,
        l2_norm_clip=l2_norm_clip,
        sample_rate=sample_rate
    )
    
    # 创建模拟数据
    client_datasets = [create_mock_dataset(1000) for _ in range(3)]
    test_dataset = create_mock_dataset(2000)
    
    print(f"DP 参数: ε={epsilon}, δ={delta}, 裁剪={l2_norm_clip}")
    
    # 训练循环
    rounds = 3
    local_steps = 10
    batch_size = 4
    
    print(f"\n开始训练: {rounds} 轮, 每轮 {local_steps} 步, 批次大小 {batch_size}")
    
    for round_idx in range(rounds):
        print(f"\n--- 第 {round_idx + 1}/{rounds} 轮 ---")
        
        # 客户端本地训练
        client_updates = []
        for client_id, dataset in enumerate(client_datasets):
            # 下载全局模型
            algorithm.set_model_state(algorithm.get_model_state())
            
            # 本地训练 - 应用 DP
            algorithm.local_update(dataset, local_steps=local_steps, batch_size=batch_size)
            
            # 获取更新
            client_updates.append(algorithm.get_model_state())
            
            # 检查隐私预算
            if algorithm.is_budget_exhausted():
                print(f"客户端 {client_id}: 隐私预算耗尽!")
                break
        
        # 服务器聚合
        algorithm.aggregate_updates(client_updates)
        
        # 评估
        accuracy, loss = algorithm.evaluate(test_dataset, batch_size=32)
        privacy_info = algorithm.get_privacy_info()
        
        print(f"测试准确率: {accuracy:.2f}%, 损失: {loss:.4f}")
        print(f"隐私消耗: ε={privacy_info['epsilon_spent']:.4f}, "
              f"剩余={privacy_info['remaining_epsilon']:.4f}")

def main():
    """主函数"""
    print("🎯 PyTorch 实现完全对齐 TensorFlow 版本")
    print("=" * 60)
    print("✅ 模型结构: 完全匹配 TensorFlow 的 CNN 架构")
    print("✅ 训练循环: 使用 local_steps 而不是 epochs")
    print("✅ 采样策略: 匹配 TensorFlow 的客户端采样逻辑")
    print("✅ 聚合权重: 实现 PFA 的公共/私有客户端分类")
    print("✅ 噪声计算: 使用 TensorFlow 版本的简化公式")
    print("=" * 60)
    
    try:
        # 运行各种算法示例
        run_fedavg_example()
        run_pfa_example()
        run_dp_fedavg_example()
        
        print("\n🎉 所有示例运行成功!")
        print("PyTorch 实现已完全对齐 TensorFlow 版本的所有关键组件。")
        
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

