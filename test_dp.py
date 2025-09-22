#!/usr/bin/env python3
"""
测试差分隐私功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.data.federated import FederatedDataSplitter
from src.models.cnn import MNISTCNN
from src.algorithms.fedavg import FedAvg
from src.algorithms.pfa import PFA
from src.algorithms.dp_pfa import DPPFA
from src.privacy.noise import DPFedAvg

def test_dp_fedavg():
    """测试带差分隐私的 FedAvg"""
    print("Testing DP-FedAvg...")
    
    # 创建模型
    model = MNISTCNN()
    device = torch.device('cpu')
    
    # 创建数据
    data_splitter = FederatedDataSplitter('mnist', num_clients=3, iid=True)
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    
    # 创建 DP-FedAvg 算法
    dp_fedavg = DPFedAvg(model, lr=0.1, device=device,
                        epsilon=2.0, delta=1e-5,
                        l2_norm_clip=1.0, sample_rate=1.0)
    
    print(f"✓ DP-FedAvg created with noise multiplier: {dp_fedavg.noise_multiplier:.4f}")
    
    # 训练几轮
    rounds = 3
    for round_idx in range(rounds):
        print(f"\n--- Round {round_idx + 1}/{rounds} ---")
        
        client_updates = []
        for dataset in client_datasets:
            dp_fedavg.set_model_state(dp_fedavg.get_model_state())
            dp_fedavg.local_update(dataset, epochs=1, batch_size=16)
            client_updates.append(dp_fedavg.get_model_state())
        
        dp_fedavg.aggregate_updates(client_updates)
        
        # 评估
        accuracy, loss = dp_fedavg.evaluate(test_dataset, batch_size=16)
        epsilon_spent, delta_spent = dp_fedavg.get_privacy_spent()
        remaining_epsilon, remaining_delta = dp_fedavg.get_remaining_budget()
        
        print(f"DP-FedAvg: Accuracy = {accuracy:.2f}%, Loss = {loss:.4f}")
        print(f"  Privacy: ε_spent = {epsilon_spent:.4f}, ε_remaining = {remaining_epsilon:.4f}")
        
        if dp_fedavg.is_budget_exhausted():
            print("  ⚠️  Privacy budget exhausted!")
            break
    
    print("DP-FedAvg test completed!")

def test_dp_pfa():
    """测试带差分隐私的 PFA"""
    print("\n" + "="*50)
    print("Testing DP-PFA...")
    
    # 创建模型
    model = MNISTCNN()
    device = torch.device('cpu')
    
    # 创建数据
    data_splitter = FederatedDataSplitter('mnist', num_clients=3, iid=True)
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    
    # 创建 DP-PFA 算法
    dp_pfa = DPPFA(model, lr=0.1, proj_dims=5, device=device,
                  epsilon=2.0, delta=1e-5,
                  l2_norm_clip=1.0, sample_rate=1.0)
    
    print(f"✓ DP-PFA created with noise multiplier: {dp_pfa.noise_multiplier:.4f}")
    print(f"✓ Projection dimensions: {dp_pfa.proj_dims}")
    
    # 训练几轮
    rounds = 3
    for round_idx in range(rounds):
        print(f"\n--- Round {round_idx + 1}/{rounds} ---")
        
        client_updates = []
        for dataset in client_datasets:
            dp_pfa.set_model_state(dp_pfa.get_model_state())
            dp_pfa.local_update(dataset, epochs=1, batch_size=16)
            client_updates.append(dp_pfa.get_model_state())
        
        dp_pfa.aggregate_updates(client_updates)
        
        # 评估
        accuracy, loss = dp_pfa.evaluate(test_dataset, batch_size=16)
        privacy_info = dp_pfa.get_privacy_info()
        
        print(f"DP-PFA: Accuracy = {accuracy:.2f}%, Loss = {loss:.4f}")
        print(f"  Privacy: ε_spent = {privacy_info['epsilon_spent']:.4f}, "
              f"ε_remaining = {privacy_info['remaining_epsilon']:.4f}")
        print(f"  Compression: {privacy_info['compression_ratio']:.2f}x")
        
        if dp_pfa.is_budget_exhausted():
            print("  ⚠️  Privacy budget exhausted!")
            break
    
    print("DP-PFA test completed!")

def compare_algorithms():
    """比较不同算法的性能"""
    print("\n" + "="*50)
    print("Comparing algorithms...")
    
    # 创建模型
    model_fedavg = MNISTCNN()
    model_pfa = MNISTCNN()
    model_dp_fedavg = MNISTCNN()
    model_dp_pfa = MNISTCNN()
    device = torch.device('cpu')
    
    # 创建数据
    data_splitter = FederatedDataSplitter('mnist', num_clients=3, iid=True)
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    
    # 创建算法
    fedavg = FedAvg(model_fedavg, lr=0.1, device=device)
    pfa = PFA(model_pfa, lr=0.1, proj_dims=5, device=device)
    dp_fedavg = DPFedAvg(model_dp_fedavg, lr=0.1, device=device,
                        epsilon=5.0, delta=1e-5, l2_norm_clip=1.0)
    dp_pfa = DPPFA(model_dp_pfa, lr=0.1, proj_dims=5, device=device,
                  epsilon=5.0, delta=1e-5, l2_norm_clip=1.0)
    
    algorithms = [
        ("FedAvg", fedavg),
        ("PFA", pfa),
        ("DP-FedAvg", dp_fedavg),
        ("DP-PFA", dp_pfa)
    ]
    
    # 训练一轮
    print("\n--- Training Round ---")
    for name, algorithm in algorithms:
        client_updates = []
        for dataset in client_datasets:
            algorithm.set_model_state(algorithm.get_model_state())
            algorithm.local_update(dataset, epochs=1, batch_size=16)
            client_updates.append(algorithm.get_model_state())
        
        algorithm.aggregate_updates(client_updates)
        
        # 评估
        accuracy, loss = algorithm.evaluate(test_dataset, batch_size=16)
        print(f"{name:10}: Accuracy = {accuracy:.2f}%, Loss = {loss:.4f}")
        
        # 显示额外信息
        if hasattr(algorithm, 'get_privacy_info'):
            privacy_info = algorithm.get_privacy_info()
            print(f"           Privacy: ε = {privacy_info['epsilon_spent']:.4f}, "
                  f"Compression = {privacy_info['compression_ratio']:.2f}x")
        elif hasattr(algorithm, 'get_compression_ratio'):
            print(f"           Compression = {algorithm.get_compression_ratio():.2f}x")
    
    print("\nAlgorithm comparison completed!")

if __name__ == '__main__':
    test_dp_fedavg()
    test_dp_pfa()
    compare_algorithms()
