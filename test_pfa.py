#!/usr/bin/env python3
"""
测试 PFA 算法的核心投影功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.data.federated import FederatedDataSplitter
from src.models.cnn import MNISTCNN
from src.algorithms.fedavg import FedAvg
from src.algorithms.pfa import PFA

def test_pfa_projection():
    """测试 PFA 投影功能"""
    print("Testing PFA projection functionality...")
    
    # 创建模型
    model = MNISTCNN()
    device = torch.device('cpu')
    
    # 创建数据
    data_splitter = FederatedDataSplitter('mnist', num_clients=3, iid=True)
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    
    print(f"✓ Data loaded: {len(client_datasets)} clients, {len(test_dataset)} test samples")
    
    # 测试不同的投影维度
    proj_dims_list = [1, 5, 10]
    
    for proj_dims in proj_dims_list:
        print(f"\n--- Testing PFA with projection dimension: {proj_dims} ---")
        
        # 创建 PFA 算法
        pfa_algorithm = PFA(model, lr=0.1, proj_dims=proj_dims, device=device)
        
        # 模拟一轮训练
        client_updates = []
        for i, dataset in enumerate(client_datasets):
            # 复制全局模型到客户端
            pfa_algorithm.set_model_state(pfa_algorithm.get_model_state())
            
            # 本地训练
            pfa_algorithm.local_update(dataset, epochs=1, batch_size=16)
            
            # 获取更新
            client_updates.append(pfa_algorithm.get_model_state())
        
        # 测试投影计算
        try:
            projection_matrix, projected_updates = pfa_algorithm.compute_projection(client_updates)
            print(f"✓ Projection computed: {projection_matrix.shape}")
            print(f"✓ Projected updates: {len(projected_updates)} vectors of shape {projected_updates[0].shape}")
            
            # 计算压缩比
            compression_ratio = pfa_algorithm.get_compression_ratio()
            print(f"✓ Compression ratio: {compression_ratio:.2f}x")
            
        except Exception as e:
            print(f"✗ Projection failed: {e}")
            continue
        
        # 测试聚合
        try:
            pfa_algorithm.aggregate_updates(client_updates)
            print("✓ PFA aggregation completed")
            
            # 评估性能
            accuracy, loss = pfa_algorithm.evaluate(test_dataset, batch_size=16)
            print(f"✓ PFA performance: Accuracy = {accuracy:.2f}%, Loss = {loss:.4f}")
            
        except Exception as e:
            print(f"✗ PFA aggregation failed: {e}")
            continue
    
    print("\nPFA projection test completed!")

def compare_fedavg_vs_pfa():
    """比较 FedAvg 和 PFA 的性能"""
    print("\n" + "="*50)
    print("Comparing FedAvg vs PFA performance...")
    
    # 创建模型
    model_fedavg = MNISTCNN()
    model_pfa = MNISTCNN()
    device = torch.device('cpu')
    
    # 创建数据
    data_splitter = FederatedDataSplitter('mnist', num_clients=5, iid=True)
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    
    # FedAvg
    fedavg_algorithm = FedAvg(model_fedavg, lr=0.1, device=device)
    
    # PFA
    pfa_algorithm = PFA(model_pfa, lr=0.1, proj_dims=10, device=device)
    
    # 训练几轮
    rounds = 3
    for round_idx in range(rounds):
        print(f"\n--- Round {round_idx + 1}/{rounds} ---")
        
        # FedAvg 训练
        fedavg_updates = []
        for dataset in client_datasets:
            fedavg_algorithm.set_model_state(fedavg_algorithm.get_model_state())
            fedavg_algorithm.local_update(dataset, epochs=1, batch_size=16)
            fedavg_updates.append(fedavg_algorithm.get_model_state())
        
        fedavg_algorithm.aggregate_updates(fedavg_updates)
        fedavg_acc, fedavg_loss = fedavg_algorithm.evaluate(test_dataset, batch_size=16)
        
        # PFA 训练
        pfa_updates = []
        for dataset in client_datasets:
            pfa_algorithm.set_model_state(pfa_algorithm.get_model_state())
            pfa_algorithm.local_update(dataset, epochs=1, batch_size=16)
            pfa_updates.append(pfa_algorithm.get_model_state())
        
        pfa_algorithm.aggregate_updates(pfa_updates)
        pfa_acc, pfa_loss = pfa_algorithm.evaluate(test_dataset, batch_size=16)
        
        print(f"FedAvg: Accuracy = {fedavg_acc:.2f}%, Loss = {fedavg_loss:.4f}")
        print(f"PFA:   Accuracy = {pfa_acc:.2f}%, Loss = {pfa_loss:.4f}")
        print(f"PFA Compression: {pfa_algorithm.get_compression_ratio():.2f}x")
    
    print("\nComparison completed!")

if __name__ == '__main__':
    test_pfa_projection()
    compare_fedavg_vs_pfa()
