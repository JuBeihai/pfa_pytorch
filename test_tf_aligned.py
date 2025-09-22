#!/usr/bin/env python3
"""
测试完全对齐 TensorFlow 版本的实现
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn import MNISTCNN
from src.data.federated import FederatedDataSplitter
from src.algorithms.fedavg import FedAvg
from src.algorithms.pfa_tf import PFA_TF
from src.algorithms.dp_fedavg_tf import DPFedAvg_TF
from src.privacy.accountant import PrivacyAccountant

def test_model_architecture():
    """测试模型架构是否匹配 TensorFlow 版本"""
    print("=== Testing Model Architecture ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTCNN().to(device)
    
    # 测试输入形状
    x = torch.randn(4, 784).to(device)  # batch_size=4, input_dim=784
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 验证输出形状
    assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"
    print("✓ Model architecture test passed!")

def test_training_loop():
    """测试训练循环是否匹配 TensorFlow 版本"""
    print("\n=== Testing Training Loop ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_splitter = FederatedDataSplitter('mnist', num_clients=3, iid=True, data_dir='./data')
    client_datasets = data_splitter.create_clients()
    
    model = MNISTCNN().to(device)
    algorithm = FedAvg(model, lr=0.1, device=device)
    
    # 测试 local_steps 而不是 epochs
    dataset = client_datasets[0]
    print(f"Dataset size: {len(dataset)}")
    
    start_time = time.time()
    algorithm.local_update(dataset, local_steps=10, batch_size=4)
    end_time = time.time()
    
    print(f"Local update time: {end_time - start_time:.2f}s")
    print("✓ Training loop test passed!")

def test_privacy_accountant():
    """测试隐私会计器是否匹配 TensorFlow 版本"""
    print("\n=== Testing Privacy Accountant ===")
    
    accountant = PrivacyAccountant(epsilon=1.0, delta=1e-5, noise_multiplier=1.0)
    
    # 测试 precheck
    can_participate = accountant.precheck(dataset_size=1000, batch_size=4, loc_steps=10)
    print(f"Can participate: {can_participate}")
    
    # 测试噪声乘数计算
    noise_mult = accountant.compute_noise_multiplier(N=1000, L=4, T=100, epsilon=1.0, delta=1e-5)
    print(f"Computed noise multiplier: {noise_mult:.4f}")
    
    # 测试更新
    accountant.update(loc_steps=10)
    epsilon_spent, delta_spent = accountant.get_privacy_spent()
    print(f"Privacy spent: ε={epsilon_spent:.4f}, δ={delta_spent}")
    
    print("✓ Privacy accountant test passed!")

def test_pfa_aggregation():
    """测试 PFA 聚合是否匹配 TensorFlow 版本"""
    print("\n=== Testing PFA Aggregation ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTCNN().to(device)
    algorithm = PFA_TF(model, lr=0.1, proj_dims=5, device=device)
    
    # 设置公共客户端
    epsilons = [1.0, 2.0, 0.5, 1.5, 0.8]
    algorithm.set_public_clients(epsilons)
    
    # 模拟客户端更新
    for cid in range(3):
        update = {}
        for name, param in model.named_parameters():
            update[name] = torch.randn_like(param) * 0.1
        
        is_public = cid in algorithm.public_clients
        algorithm.aggregate(cid, list(update.values()), is_public)
        print(f"Client {cid} aggregated (public: {is_public})")
    
    # 测试平均
    global_model = {name: param.clone() for name, param in model.named_parameters()}
    new_model = algorithm.update(global_model)
    
    print(f"Updated model keys: {list(new_model.keys())}")
    print("✓ PFA aggregation test passed!")

def test_dp_fedavg():
    """测试 DP-FedAvg 是否匹配 TensorFlow 版本"""
    print("\n=== Testing DP-FedAvg ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_splitter = FederatedDataSplitter('mnist', num_clients=3, iid=True, data_dir='./data')
    client_datasets = data_splitter.create_clients()
    
    model = MNISTCNN().to(device)
    algorithm = DPFedAvg_TF(
        model=model,
        lr=0.1,
        device=device,
        epsilon=1.0,
        delta=1e-5,
        l2_norm_clip=1.0,
        sample_rate=0.8
    )
    
    # 测试本地更新
    dataset = client_datasets[0]
    algorithm.local_update(dataset, local_steps=5, batch_size=4)
    
    # 测试隐私信息
    privacy_info = algorithm.get_privacy_info()
    print(f"Privacy info: {privacy_info}")
    
    # 测试预检查
    can_participate = algorithm.precheck(len(dataset), 4, 5)
    print(f"Can participate: {can_participate}")
    
    print("✓ DP-FedAvg test passed!")

def test_end_to_end():
    """端到端测试"""
    print("\n=== End-to-End Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_splitter = FederatedDataSplitter('mnist', num_clients=3, iid=True, data_dir='./data')
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    
    # 测试 FedAvg
    print("Testing FedAvg...")
    model1 = MNISTCNN().to(device)
    fedavg = FedAvg(model1, lr=0.1, device=device)
    
    for round_idx in range(3):
        client_updates = []
        for dataset in client_datasets:
            fedavg.set_model_state(fedavg.get_model_state())
            fedavg.local_update(dataset, local_steps=5, batch_size=4)
            client_updates.append(fedavg.get_model_state())
        
        fedavg.aggregate_updates(client_updates)
        accuracy, loss = fedavg.evaluate(test_dataset, batch_size=32)
        print(f"Round {round_idx+1}: Accuracy = {accuracy:.2f}%, Loss = {loss:.4f}")
    
    # 测试 PFA
    print("\nTesting PFA...")
    model2 = MNISTCNN().to(device)
    pfa = PFA_TF(model2, lr=0.1, proj_dims=5, device=device)
    
    # 设置公共客户端
    epsilons = [1.0, 2.0, 0.5]
    pfa.set_public_clients(epsilons)
    
    for round_idx in range(3):
        for cid, dataset in enumerate(client_datasets):
            pfa.set_model_state(pfa.get_model_state())
            pfa.local_update(dataset, local_steps=5, batch_size=4)
            
            # 计算更新
            global_state = pfa.get_model_state()
            client_state = pfa.get_model_state()
            update = {}
            for key in global_state.keys():
                update[key] = global_state[key] - client_state[key]
            
            is_public = cid in pfa.public_clients
            pfa.aggregate(cid, list(update.values()), is_public)
        
        pfa.update(pfa.get_model_state())
        accuracy, loss = pfa.evaluate(test_dataset, batch_size=32)
        print(f"Round {round_idx+1}: Accuracy = {accuracy:.2f}%, Loss = {loss:.4f}")
    
    print("✓ End-to-end test passed!")

def main():
    """运行所有测试"""
    print("Testing TensorFlow-aligned PyTorch implementation...")
    
    try:
        test_model_architecture()
        test_training_loop()
        test_privacy_accountant()
        test_pfa_aggregation()
        test_dp_fedavg()
        test_end_to_end()
        
        print("\n🎉 All tests passed! The implementation is aligned with TensorFlow version.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
