#!/usr/bin/env python3
"""
测试优化后的差分隐私功能
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

def test_optimized_dp_parameters():
    """测试优化后的差分隐私参数"""
    print("Testing optimized DP parameters...")
    
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
            'name': 'Conservative DP',
            'epsilon': 10.0,
            'delta': 1e-5,
            'l2_norm_clip': 5.0,
            'sample_rate': 0.1,
            'batch_size': 64
        },
        {
            'name': 'Moderate DP',
            'epsilon': 5.0,
            'delta': 1e-5,
            'l2_norm_clip': 3.0,
            'sample_rate': 0.2,
            'batch_size': 32
        },
        {
            'name': 'Aggressive DP',
            'epsilon': 2.0,
            'delta': 1e-5,
            'l2_norm_clip': 2.0,
            'sample_rate': 0.3,
            'batch_size': 16
        }
    ]
    
    for config in test_configs:
        print(f"\n--- Testing {config['name']} ---")
        print(f"Parameters: ε={config['epsilon']}, δ={config['delta']}, "
              f"clip={config['l2_norm_clip']}, sample_rate={config['sample_rate']}")
        
        # 创建 DP-PFA 算法
        dp_pfa = DPPFA(model, lr=0.01, proj_dims=10, device=device,
                      epsilon=config['epsilon'], delta=config['delta'],
                      l2_norm_clip=config['l2_norm_clip'],
                      sample_rate=config['sample_rate'])
        
        print(f"Noise multiplier: {dp_pfa.noise_multiplier:.4f}")
        
        # 训练几轮
        rounds = 5
        for round_idx in range(rounds):
            client_updates = []
            for dataset in client_datasets:
                dp_pfa.set_model_state(dp_pfa.get_model_state())
                dp_pfa.local_update(dataset, epochs=1, batch_size=config['batch_size'])
                client_updates.append(dp_pfa.get_model_state())
            
            dp_pfa.aggregate_updates(client_updates)
            
            # 评估
            accuracy, loss = dp_pfa.evaluate(test_dataset, batch_size=config['batch_size'])
            privacy_info = dp_pfa.get_privacy_info()
            
            print(f"Round {round_idx + 1}: Accuracy = {accuracy:.2f}%, Loss = {loss:.4f}")
            print(f"  Privacy: ε_spent = {privacy_info['epsilon_spent']:.4f}, "
                  f"ε_remaining = {privacy_info['remaining_epsilon']:.4f}")
            
            if dp_pfa.is_budget_exhausted():
                print("  ⚠️  Privacy budget exhausted!")
                break
        
        print(f"Final: Accuracy = {accuracy:.2f}%, ε_spent = {privacy_info['epsilon_spent']:.4f}")

def test_dp_vs_non_dp():
    """对比差分隐私和非差分隐私的性能"""
    print("\n" + "="*60)
    print("Comparing DP vs Non-DP performance...")
    
    # 创建模型
    model_pfa = MNISTCNN()
    model_dp_pfa = MNISTCNN()
    device = torch.device('cpu')
    
    # 创建数据
    data_splitter = FederatedDataSplitter('mnist', num_clients=5, iid=True)
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    
    # 创建算法
    pfa = PFA(model_pfa, lr=0.01, proj_dims=10, device=device)
    dp_pfa = DPPFA(model_dp_pfa, lr=0.01, proj_dims=10, device=device,
                  epsilon=10.0, delta=1e-5, l2_norm_clip=5.0, sample_rate=0.1)
    
    print(f"PFA noise multiplier: {dp_pfa.noise_multiplier:.4f}")
    
    # 训练多轮
    rounds = 10
    for round_idx in range(rounds):
        print(f"\n--- Round {round_idx + 1}/{rounds} ---")
        
        # PFA 训练
        pfa_updates = []
        for dataset in client_datasets:
            pfa.set_model_state(pfa.get_model_state())
            pfa.local_update(dataset, epochs=1, batch_size=64)
            pfa_updates.append(pfa.get_model_state())
        
        pfa.aggregate_updates(pfa_updates)
        pfa_acc, pfa_loss = pfa.evaluate(test_dataset, batch_size=64)
        
        # DP-PFA 训练
        dp_pfa_updates = []
        for dataset in client_datasets:
            dp_pfa.set_model_state(dp_pfa.get_model_state())
            dp_pfa.local_update(dataset, epochs=1, batch_size=64)
            dp_pfa_updates.append(dp_pfa.get_model_state())
        
        dp_pfa.aggregate_updates(dp_pfa_updates)
        dp_pfa_acc, dp_pfa_loss = dp_pfa.evaluate(test_dataset, batch_size=64)
        
        print(f"PFA:    Accuracy = {pfa_acc:.2f}%, Loss = {pfa_loss:.4f}")
        print(f"DP-PFA: Accuracy = {dp_pfa_acc:.2f}%, Loss = {dp_pfa_loss:.4f}")
        
        if hasattr(dp_pfa, 'get_privacy_info'):
            privacy_info = dp_pfa.get_privacy_info()
            print(f"        Privacy: ε = {privacy_info['epsilon_spent']:.4f}")
        
        if dp_pfa.is_budget_exhausted():
            print("⚠️  Privacy budget exhausted!")
            break

def test_different_epsilon_values():
    """测试不同 epsilon 值的影响"""
    print("\n" + "="*60)
    print("Testing different epsilon values...")
    
    epsilon_values = [1.0, 2.0, 5.0, 10.0, 20.0]
    
    for epsilon in epsilon_values:
        print(f"\n--- Testing ε = {epsilon} ---")
        
        # 创建模型
        model = MNISTCNN()
        device = torch.device('cpu')
        
        # 创建数据
        data_splitter = FederatedDataSplitter('mnist', num_clients=3, iid=True)
        client_datasets = data_splitter.create_clients()
        test_dataset = data_splitter.get_test_dataset()
        
        # 创建 DP-PFA 算法
        dp_pfa = DPPFA(model, lr=0.01, proj_dims=10, device=device,
                      epsilon=epsilon, delta=1e-5, l2_norm_clip=3.0, sample_rate=0.2)
        
        print(f"Noise multiplier: {dp_pfa.noise_multiplier:.4f}")
        
        # 训练几轮
        rounds = 3
        for round_idx in range(rounds):
            client_updates = []
            for dataset in client_datasets:
                dp_pfa.set_model_state(dp_pfa.get_model_state())
                dp_pfa.local_update(dataset, epochs=1, batch_size=32)
                client_updates.append(dp_pfa.get_model_state())
            
            dp_pfa.aggregate_updates(client_updates)
            
            if dp_pfa.is_budget_exhausted():
                print(f"  Privacy budget exhausted at round {round_idx + 1}")
                break
        
        # 最终评估
        accuracy, loss = dp_pfa.evaluate(test_dataset, batch_size=32)
        privacy_info = dp_pfa.get_privacy_info()
        
        print(f"Final: Accuracy = {accuracy:.2f}%, ε_spent = {privacy_info['epsilon_spent']:.4f}")

if __name__ == '__main__':
    test_optimized_dp_parameters()
    test_dp_vs_non_dp()
    test_different_epsilon_values()
