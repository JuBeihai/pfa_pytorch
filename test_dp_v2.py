#!/usr/bin/env python3
"""
测试改进的差分隐私功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.data.federated import FederatedDataSplitter
from src.models.cnn import MNISTCNN
from src.algorithms.fedavg import FedAvg
from src.algorithms.pfa import PFA
from src.algorithms.dp_pfa_v2 import DPPFAV2

def test_improved_dp_pfa():
    """测试改进的 DP-PFA"""
    print("Testing improved DP-PFA...")
    
    # 创建模型
    model = MNISTCNN()
    device = torch.device('cpu')
    
    # 创建数据
    data_splitter = FederatedDataSplitter('mnist', num_clients=5, iid=True)
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    
    # 测试不同的参数组合
    test_configs = [
        {
            'name': 'Light DP',
            'epsilon': 5.0,
            'noise_multiplier': 0.5,
            'l2_norm_clip': 2.0,
            'proj_dims': 10
        },
        {
            'name': 'Medium DP',
            'epsilon': 3.0,
            'noise_multiplier': 0.3,
            'l2_norm_clip': 1.5,
            'proj_dims': 15
        },
        {
            'name': 'Strong DP',
            'epsilon': 1.0,
            'noise_multiplier': 0.1,
            'l2_norm_clip': 1.0,
            'proj_dims': 20
        }
    ]
    
    for config in test_configs:
        print(f"\n--- Testing {config['name']} ---")
        print(f"Parameters: ε={config['epsilon']}, noise={config['noise_multiplier']}, "
              f"clip={config['l2_norm_clip']}, proj_dims={config['proj_dims']}")
        
        # 创建改进的 DP-PFA 算法
        dp_pfa = DPPFAV2(model, lr=0.01, proj_dims=config['proj_dims'], device=device,
                        epsilon=config['epsilon'], delta=1e-5,
                        noise_multiplier=config['noise_multiplier'],
                        l2_norm_clip=config['l2_norm_clip'])
        
        print(f"Actual noise multiplier: {dp_pfa.noise_multiplier:.4f}")
        
        # 训练多轮
        rounds = 10
        for round_idx in range(rounds):
            client_updates = []
            for dataset in client_datasets:
                dp_pfa.set_model_state(dp_pfa.get_model_state())
                dp_pfa.local_update(dataset, epochs=1, batch_size=32)
                client_updates.append(dp_pfa.get_model_state())
            
            dp_pfa.aggregate_updates(client_updates)
            
            # 评估
            accuracy, loss = dp_pfa.evaluate(test_dataset, batch_size=32)
            privacy_info = dp_pfa.get_privacy_info()
            
            if (round_idx + 1) % 2 == 0 or round_idx == 0:
                print(f"Round {round_idx + 1}: Accuracy = {accuracy:.2f}%, Loss = {loss:.4f}")
                print(f"  Privacy: ε_spent = {privacy_info['epsilon_spent']:.4f}, "
                      f"ε_remaining = {privacy_info['remaining_epsilon']:.4f}")
            
            if dp_pfa.is_budget_exhausted():
                print(f"  ⚠️  Privacy budget exhausted at round {round_idx + 1}")
                break
        
        print(f"Final: Accuracy = {accuracy:.2f}%, ε_spent = {privacy_info['epsilon_spent']:.4f}")

def compare_all_algorithms():
    """对比所有算法"""
    print("\n" + "="*60)
    print("Comparing all algorithms...")
    
    # 创建模型
    model_fedavg = MNISTCNN()
    model_pfa = MNISTCNN()
    model_dp_pfa = MNISTCNN()
    device = torch.device('cpu')
    
    # 创建数据
    data_splitter = FederatedDataSplitter('mnist', num_clients=5, iid=True)
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    
    # 创建算法
    fedavg = FedAvg(model_fedavg, lr=0.01, device=device)
    pfa = PFA(model_pfa, lr=0.01, proj_dims=10, device=device)
    dp_pfa = DPPFAV2(model_dp_pfa, lr=0.01, proj_dims=10, device=device,
                    epsilon=5.0, noise_multiplier=0.5, l2_norm_clip=2.0)
    
    algorithms = [
        ("FedAvg", fedavg),
        ("PFA", pfa),
        ("DP-PFA V2", dp_pfa)
    ]
    
    # 训练多轮
    rounds = 8
    for round_idx in range(rounds):
        print(f"\n--- Round {round_idx + 1}/{rounds} ---")
        
        for name, algorithm in algorithms:
            client_updates = []
            for dataset in client_datasets:
                algorithm.set_model_state(algorithm.get_model_state())
                algorithm.local_update(dataset, epochs=1, batch_size=32)
                client_updates.append(algorithm.get_model_state())
            
            algorithm.aggregate_updates(client_updates)
            
            # 评估
            accuracy, loss = algorithm.evaluate(test_dataset, batch_size=32)
            print(f"{name:12}: Accuracy = {accuracy:.2f}%, Loss = {loss:.4f}")
            
            # 显示额外信息
            if hasattr(algorithm, 'get_privacy_info'):
                privacy_info = algorithm.get_privacy_info()
                print(f"             Privacy: ε = {privacy_info['epsilon_spent']:.4f}, "
                      f"Compression = {privacy_info['compression_ratio']:.2f}x")
            elif hasattr(algorithm, 'get_compression_ratio'):
                print(f"             Compression = {algorithm.get_compression_ratio():.2f}x")
    
    print("\nAlgorithm comparison completed!")

def test_privacy_utility_tradeoff():
    """测试隐私-效用权衡"""
    print("\n" + "="*60)
    print("Testing privacy-utility tradeoff...")
    
    noise_multipliers = [0.1, 0.3, 0.5, 0.8, 1.0]
    
    for noise_mult in noise_multipliers:
        print(f"\n--- Testing noise multiplier: {noise_mult} ---")
        
        # 创建模型
        model = MNISTCNN()
        device = torch.device('cpu')
        
        # 创建数据
        data_splitter = FederatedDataSplitter('mnist', num_clients=3, iid=True)
        client_datasets = data_splitter.create_clients()
        test_dataset = data_splitter.get_test_dataset()
        
        # 创建 DP-PFA 算法
        dp_pfa = DPPFAV2(model, lr=0.01, proj_dims=10, device=device,
                        epsilon=10.0, noise_multiplier=noise_mult, l2_norm_clip=2.0)
        
        # 训练几轮
        rounds = 5
        for round_idx in range(rounds):
            client_updates = []
            for dataset in client_datasets:
                dp_pfa.set_model_state(dp_pfa.get_model_state())
                dp_pfa.local_update(dataset, epochs=1, batch_size=32)
                client_updates.append(dp_pfa.get_model_state())
            
            dp_pfa.aggregate_updates(client_updates)
        
        # 最终评估
        accuracy, loss = dp_pfa.evaluate(test_dataset, batch_size=32)
        privacy_info = dp_pfa.get_privacy_info()
        
        print(f"Final: Accuracy = {accuracy:.2f}%, Loss = {loss:.4f}")
        print(f"        Privacy: ε = {privacy_info['epsilon_spent']:.4f}")

if __name__ == '__main__':
    test_improved_dp_pfa()
    compare_all_algorithms()
    test_privacy_utility_tradeoff()
