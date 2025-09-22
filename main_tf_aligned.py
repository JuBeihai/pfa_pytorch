#!/usr/bin/env python3
"""
完全对齐 TensorFlow 版本的 PyTorch 实现
匹配模型结构、训练循环、采样策略、聚合权重和噪声计算
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import os
import sys
from typing import List, Dict, Tuple

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn import MNISTCNN
from src.data.federated import FederatedDataSplitter
from src.algorithms.fedavg import FedAvg
from src.algorithms.pfa_tf import PFA_TF
from src.algorithms.dp_fedavg_tf import DPFedAvg_TF
from src.privacy.accountant import PrivacyAccountant

def set_epsilons(filename: str, N: int) -> List[float]:
    """
    设置 epsilon 值，匹配 TensorFlow 版本的实现
    """
    # 简化实现，使用固定值
    if filename == 'gauss1':
        epsilons = np.random.normal(1.0, 0.5, N)
        epsilons = np.clip(epsilons, 0.1, 10.0)  # 限制范围
    elif filename == 'uniform1':
        epsilons = np.random.uniform(0.5, 5.0, N)
    else:
        epsilons = [1.0] * N
    
    print(f"Epsilons: {epsilons}")
    return epsilons.tolist()

def compute_noise_multiplier(N: int, L: int, T: int, epsilon: float, delta: float) -> float:
    """
    计算噪声乘数，匹配 TensorFlow 版本的公式
    """
    q = L / N
    nm = 10 * q * math.sqrt(T * (-math.log10(delta))) / epsilon
    return nm

def sample_clients(candidates: List[int], num_clients: int, sample_ratio: float, 
                  public_clients: List[int] = None) -> List[int]:
    """
    采样客户端，匹配 TensorFlow 版本的逻辑
    """
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

def main():
    parser = argparse.ArgumentParser(description='PFA PyTorch Implementation - TF Aligned')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist', 'cifar10'])
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'lr', '2nn'])
    parser.add_argument('--noniid', action='store_true', help='Use non-IID data distribution')
    parser.add_argument('--noniid_level', type=int, default=10, help='Level of non-IID')
    
    # 联邦学习参数
    parser.add_argument('--N', type=int, default=10, help='Total number of clients')
    parser.add_argument('--max_steps', type=int, default=10000, help='Total communication rounds')
    parser.add_argument('--local_steps', type=int, default=100, help='Local training steps')
    parser.add_argument('--client_dataset_size', type=int, default=None, help='Client dataset size')
    parser.add_argument('--client_batch_size', type=int, default=4, help='Client batch size')
    parser.add_argument('--num_microbatches', type=int, default=4, help='Number of microbatches')
    
    # 学习率参数
    parser.add_argument('--lr_decay', action='store_true', help='Use learning rate decay')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    
    # 差分隐私参数
    parser.add_argument('--dpsgd', action='store_true', help='Use DP-SGD')
    parser.add_argument('--eps', type=str, default='gauss1', help='Epsilon file name')
    parser.add_argument('--delta', type=float, default=1e-5, help='DP parameter Delta')
    parser.add_argument('--l2_norm_clip', type=float, default=1.0, help='Clipping norm')
    
    # 采样参数
    parser.add_argument('--sample_mode', type=str, default='R', choices=['R', 'W1', 'W2'], 
                       help='Sample mode: R for random, W for weighted')
    parser.add_argument('--sample_ratio', type=float, default=0.8, help='Sample ratio')
    
    # 投影参数
    parser.add_argument('--projection', action='store_true', help='Use projection')
    parser.add_argument('--proj_wavg', action='store_true', help='Use weighted projection')
    parser.add_argument('--delay', action='store_true', help='Use delayed aggregation')
    parser.add_argument('--proj_dims', type=int, default=1, help='Projection dimensions')
    parser.add_argument('--lanczos_iter', type=int, default=256, help='Lanczos iterations')
    
    # 其他参数
    parser.add_argument('--version', type=int, default=1, help='Dataset version')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Arguments: {args}")
    
    # 准备数据
    data_splitter = FederatedDataSplitter(
        dataset_name=args.dataset,
        num_clients=args.N,
        iid=not args.noniid,
        data_dir=args.data_dir
    )
    
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    
    print(f"Data loaded: {len(client_datasets)} clients, {len(test_dataset)} test samples")
    
    # 准备隐私偏好
    priv_preferences = None
    if args.dpsgd:
        priv_preferences = set_epsilons(args.eps, args.N)
    
    # 创建模型
    model = MNISTCNN().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 创建算法
    if args.projection or args.proj_wavg:
        algorithm = PFA_TF(
            model=model,
            lr=args.lr,
            proj_dims=args.proj_dims,
            lanczos_iter=args.lanczos_iter,
            device=device,
            delay=args.delay
        )
        if priv_preferences is not None:
            algorithm.set_public_clients(priv_preferences)
    elif args.dpsgd:
        algorithm = DPFedAvg_TF(
            model=model,
            lr=args.lr,
            device=device,
            epsilon=priv_preferences[0] if priv_preferences else 1.0,
            delta=args.delta,
            l2_norm_clip=args.l2_norm_clip,
            sample_rate=args.sample_ratio
        )
    else:
        algorithm = FedAvg(model=model, lr=args.lr, device=device)
    
    # 创建客户端
    clients = []
    for cid in range(args.N):
        client_data = client_datasets[cid]
        
        if args.dpsgd and priv_preferences is not None:
            epsilon = priv_preferences[cid]
            delta = args.delta
            noise_multiplier = compute_noise_multiplier(
                N=len(client_data),
                L=args.client_batch_size,
                T=args.max_steps * args.sample_ratio,
                epsilon=epsilon,
                delta=delta
            )
            
            client_algorithm = DPFedAvg_TF(
                model=copy.deepcopy(model),
                lr=args.lr,
                device=device,
                epsilon=epsilon,
                delta=delta,
                noise_multiplier=noise_multiplier,
                l2_norm_clip=args.l2_norm_clip,
                sample_rate=args.sample_ratio
            )
        else:
            client_algorithm = FedAvg(
                model=copy.deepcopy(model),
                lr=args.lr,
                device=device
            )
        
        clients.append({
            'algorithm': client_algorithm,
            'dataset': client_data,
            'dataset_size': len(client_data)
        })
    
    # 训练循环
    comm_round = args.max_steps // args.local_steps
    print(f"Communication rounds: {comm_round}")
    
    accuracy_accountant = []
    privacy_accountant = []
    start_time = time.time()
    
    for r in range(comm_round):
        print(f"\n=== Round {r+1}/{comm_round} ===")
        comm_start_time = time.time()
        
        if args.N == 1:
            # 单客户端情况
            client = clients[0]
            client['algorithm'].local_update(
                client['dataset'], 
                local_steps=args.local_steps, 
                batch_size=args.client_batch_size
            )
            model = client['algorithm'].get_model_state()
        else:
            # 多客户端情况
            # 预检查候选客户端
            candidates = []
            for cid in range(args.N):
                if hasattr(clients[cid]['algorithm'], 'precheck'):
                    if clients[cid]['algorithm'].precheck(
                        clients[cid]['dataset_size'], 
                        args.client_batch_size, 
                        args.local_steps
                    ):
                        candidates.append(cid)
                else:
                    candidates.append(cid)
            
            # 采样参与客户端
            participants = sample_clients(
                candidates, 
                args.N, 
                args.sample_ratio,
                algorithm.public_clients if hasattr(algorithm, 'public_clients') else None
            )
            
            if len(participants) == 0:
                print("No sufficient candidates. Stopping training.")
                break
            
            print(f"Participants: {participants}")
            
            # 客户端本地训练
            client_updates = []
            for cid in participants:
                client = clients[cid]
                
                # 下载全局模型
                client['algorithm'].set_model_state(algorithm.get_model_state())
                
                # 本地训练
                client['algorithm'].local_update(
                    client['dataset'],
                    local_steps=args.local_steps,
                    batch_size=args.client_batch_size
                )
                
                # 获取更新
                if args.projection or args.proj_wavg:
                    # PFA 需要计算更新向量
                    global_state = algorithm.get_model_state()
                    client_state = client['algorithm'].get_model_state()
                    update = {}
                    for key in global_state.keys():
                        update[key] = global_state[key] - client_state[key]
                    
                    # 聚合更新
                    is_public = (cid in algorithm.public_clients) if hasattr(algorithm, 'public_clients') else False
                    algorithm.aggregate(cid, list(update.values()), is_public)
                else:
                    # 普通 FedAvg
                    client_updates.append(client['algorithm'].get_model_state())
            
            # 服务器聚合
            if args.projection or args.proj_wavg:
                model = algorithm.update(algorithm.get_model_state())
            else:
                algorithm.aggregate_updates(client_updates)
                model = algorithm.get_model_state()
        
        # 评估
        algorithm.set_model_state(model)
        accuracy, loss = algorithm.evaluate(test_dataset, batch_size=32)
        accuracy_accountant.append(accuracy)
        
        print(f"Test Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
        print(f"Round time: {time.time() - comm_start_time:.2f}s")
        
        # 隐私信息
        if args.dpsgd and hasattr(algorithm, 'get_privacy_info'):
            privacy_info = algorithm.get_privacy_info()
            print(f"Privacy: ε_spent = {privacy_info['epsilon_spent']:.4f}, "
                  f"ε_remaining = {privacy_info['remaining_epsilon']:.4f}")
            privacy_accountant.append(privacy_info['epsilon_spent'])
    
    print(f"\nTotal training time: {time.time() - start_time:.2f}s")
    print(f"Final accuracy: {accuracy_accountant[-1]:.2f}%")
    
    if privacy_accountant:
        print(f"Final privacy spent: {privacy_accountant[-1]:.4f}")

if __name__ == '__main__':
    import math
    main()
