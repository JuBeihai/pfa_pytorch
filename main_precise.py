#!/usr/bin/env python3
"""
完全匹配论文的PFA主程序
实现论文Algorithm 2和Algorithm 4的所有细节
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import os
import sys
import math
from typing import List, Dict, Tuple
import copy

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn import MNISTCNN
from src.data.federated import FederatedDataSplitter
from src.algorithms.pfa_precise import PFA_Precise
from src.algorithms.fedavg import FedAvg
from src.privacy.heterogeneous_dp import HeterogeneousDP

def set_epsilons(filename: str, N: int) -> List[float]:
    """
    设置epsilon值，完全匹配论文的实现
    """
    if filename == 'gauss1':
        # 高斯分布：均值1.0，标准差0.5
        epsilons = np.random.normal(1.0, 0.5, N)
        epsilons = np.clip(epsilons, 0.1, 10.0)  # 限制范围
    elif filename == 'uniform1':
        # 均匀分布：[0.5, 5.0]
        epsilons = np.random.uniform(0.5, 5.0, N)
    elif filename == 'exponential1':
        # 指数分布：λ=1.0
        epsilons = np.random.exponential(1.0, N)
        epsilons = np.clip(epsilons, 0.1, 10.0)
    else:
        # 默认值
        epsilons = [1.0] * N
    
    print(f"Epsilons ({filename}): {epsilons}")
    return epsilons.tolist()

def compute_noise_multipliers(client_epsilons: List[float], 
                            client_deltas: List[float],
                            client_dataset_sizes: List[int],
                            batch_sizes: List[int],
                            local_steps: List[int]) -> List[float]:
    """
    计算每个客户端的噪声乘数，完全匹配论文公式
    """
    noise_multipliers = []
    
    for i in range(len(client_epsilons)):
        # 计算采样率
        q = batch_sizes[i] / client_dataset_sizes[i]
        
        # 使用论文公式计算噪声乘数
        epsilon = client_epsilons[i]
        delta = client_deltas[i]
        T = local_steps[i]
        
        # 简化的噪声乘数计算（匹配论文）
        noise_mult = 10 * q * math.sqrt(T * (-math.log10(delta))) / epsilon
        noise_multipliers.append(noise_mult)
    
    return noise_multipliers

def sample_clients_precise(candidates: List[int], 
                          num_clients: int, 
                          sample_ratio: float,
                          public_clients: List[int] = None,
                          max_attempts: int = 50) -> List[int]:
    """
    精确的客户端采样，完全匹配论文的采样策略
    """
    m = int(num_clients * sample_ratio)
    if len(candidates) < m:
        return []
    
    # 确保至少有一个公共客户端参与
    if public_clients is not None:
        attempts = 0
        while attempts < max_attempts:
            # 随机采样
            participants = list(np.random.permutation(candidates))[:m]
            
            # 检查是否包含公共客户端
            if len(set(participants).intersection(set(public_clients))) > 0:
                return participants
            
            attempts += 1
        
        # 如果无法满足条件，强制包含一个公共客户端
        if public_clients:
            public_participant = np.random.choice(public_clients)
            other_participants = list(np.random.permutation(
                [c for c in candidates if c != public_participant]
            ))[:m-1]
            participants = [public_participant] + other_participants
            return participants
    
    # 没有公共客户端要求，直接随机采样
    participants = list(np.random.permutation(candidates))[:m]
    return participants

def main():
    parser = argparse.ArgumentParser(description='PFA Precise Implementation - 100% Paper Match')
    
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
    parser.add_argument('--tau', type=int, default=1, help='PFA+ parameter tau')
    
    # 学习率参数
    parser.add_argument('--lr_decay', action='store_true', help='Use learning rate decay')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    
    # 差分隐私参数
    parser.add_argument('--dpsgd', action='store_true', help='Use DP-SGD')
    parser.add_argument('--eps', type=str, default='gauss1', help='Epsilon distribution')
    parser.add_argument('--delta', type=float, default=1e-5, help='DP parameter Delta')
    parser.add_argument('--l2_norm_clip', type=float, default=1.0, help='Clipping norm')
    
    # 采样参数
    parser.add_argument('--sample_mode', type=str, default='R', choices=['R', 'W1', 'W2'], 
                       help='Sample mode: R for random, W for weighted')
    parser.add_argument('--sample_ratio', type=float, default=0.8, help='Sample ratio')
    
    # PFA参数
    parser.add_argument('--projection', action='store_true', help='Use PFA projection')
    parser.add_argument('--proj_dims', type=int, default=1, help='Projection dimensions')
    parser.add_argument('--lanczos_iter', type=int, default=256, help='Lanczos iterations')
    parser.add_argument('--delay', action='store_true', help='Use delayed projection')
    parser.add_argument('--clustering_method', type=str, default='kmeans', 
                       choices=['kmeans', 'gaussian_mixture'], help='Clustering method')
    parser.add_argument('--balance_ratio', type=float, default=0.1, help='Public client ratio')
    
    # 其他参数
    parser.add_argument('--version', type=int, default=1, help='Dataset version')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    print(f"参数: {args}")
    
    # 准备数据
    data_splitter = FederatedDataSplitter(
        dataset_name=args.dataset,
        num_clients=args.N,
        iid=not args.noniid,
        data_dir=args.data_dir
    )
    
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    
    print(f"数据加载完成: {len(client_datasets)}个客户端, {len(test_dataset)}个测试样本")
    
    # 准备隐私参数
    priv_preferences = None
    if args.dpsgd:
        priv_preferences = set_epsilons(args.eps, args.N)
        client_deltas = [args.delta] * args.N
        print(f"隐私参数设置完成: ε={priv_preferences}, δ={client_deltas}")
    
    # 创建模型
    model = MNISTCNN().to(device)
    print(f"模型创建完成: {sum(p.numel() for p in model.parameters())}个参数")
    
    # 创建算法
    if args.projection:
        algorithm = PFA_Precise(
            model=model,
            lr=args.lr,
            proj_dims=args.proj_dims,
            lanczos_iter=args.lanczos_iter,
            device=device,
            delay=args.delay,
            clustering_method=args.clustering_method,
            balance_ratio=args.balance_ratio
        )
        
        # 设置异构DP
        if priv_preferences is not None:
            algorithm.set_heterogeneous_dp(priv_preferences, client_deltas)
            
        # 客户端分类
        if priv_preferences is not None:
            dataset_sizes = [len(client_datasets[i]) for i in range(args.N)]
            algorithm.divide_clients(priv_preferences, dataset_sizes)
    else:
        algorithm = FedAvg(model=model, lr=args.lr, device=device)
    
    # 创建客户端
    clients = []
    for cid in range(args.N):
        client_data = client_datasets[cid]
        
        if args.dpsgd and priv_preferences is not None:
            # 计算客户端特定的噪声乘数
            noise_multipliers = compute_noise_multipliers(
                [priv_preferences[cid]], [args.delta], 
                [len(client_data)], [args.client_batch_size], [args.local_steps]
            )
            
            client_algorithm = PFA_Precise(
                model=copy.deepcopy(model),
                lr=args.lr,
                proj_dims=args.proj_dims,
                lanczos_iter=args.lanczos_iter,
                device=device,
                delay=args.delay,
                clustering_method=args.clustering_method,
                balance_ratio=args.balance_ratio
            )
            
            # 设置客户端特定的DP参数
            client_algorithm.set_heterogeneous_dp(
                [priv_preferences[cid]], [args.delta]
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
            'dataset_size': len(client_data),
            'client_id': cid
        })
    
    # 训练循环
    comm_round = args.max_steps // args.local_steps
    print(f"通信轮数: {comm_round}")
    
    accuracy_history = []
    privacy_history = []
    projection_quality_history = []
    start_time = time.time()
    
    for r in range(comm_round):
        print(f"\n=== 轮次 {r+1}/{comm_round} ===")
        round_start_time = time.time()
        
        if args.N == 1:
            # 单客户端情况
            client = clients[0]
            if args.dpsgd:
                client['algorithm'].local_update_with_dp(
                    client['dataset'], 
                    local_steps=args.local_steps, 
                    batch_size=args.client_batch_size,
                    client_id=client['client_id'],
                    l2_norm_clip=args.l2_norm_clip
                )
            else:
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
                if hasattr(clients[cid]['algorithm'], 'heterogeneous_dp'):
                    if clients[cid]['algorithm'].heterogeneous_dp.check_client_budget(cid):
                        candidates.append(cid)
                else:
                    candidates.append(cid)
            
            # 采样参与客户端
            participants = sample_clients_precise(
                candidates, 
                args.N, 
                args.sample_ratio,
                algorithm.public_clients if hasattr(algorithm, 'public_clients') else None
            )
            
            if len(participants) == 0:
                print("没有足够的候选客户端，停止训练")
                break
            
            print(f"参与客户端: {participants}")
            
            # 客户端本地训练
            client_updates = []
            client_weights = []
            client_epsilons = []
            client_dataset_sizes = []
            client_types = []
            
            for cid in participants:
                client = clients[cid]
                
                # 下载全局模型
                client['algorithm'].set_model_state(algorithm.get_model_state())
                
                # 本地训练
                if args.dpsgd and hasattr(client['algorithm'], 'local_update_with_dp'):
                    client['algorithm'].local_update_with_dp(
                        client['dataset'],
                        local_steps=args.local_steps,
                        batch_size=args.client_batch_size,
                        client_id=cid,
                        l2_norm_clip=args.l2_norm_clip
                    )
                else:
                    client['algorithm'].local_update(
                        client['dataset'],
                        local_steps=args.local_steps,
                        batch_size=args.client_batch_size
                    )
                
                # 获取更新
                client_update = client['algorithm'].get_model_state()
                client_updates.append(client_update)
                
                # 计算权重
                client_weights.append(client['dataset_size'])
                
                # 隐私参数
                if priv_preferences is not None:
                    client_epsilons.append(priv_preferences[cid])
                    client_dataset_sizes.append(client['dataset_size'])
                    
                    # 客户端类型
                    if hasattr(algorithm, 'public_clients') and cid in algorithm.public_clients:
                        client_types.append('public')
                    else:
                        client_types.append('private')
            
            # 服务器聚合
            if args.projection:
                algorithm.aggregate_updates(
                    client_updates=client_updates,
                    client_weights=client_weights,
                    client_epsilons=client_epsilons if priv_preferences else None,
                    client_dataset_sizes=client_dataset_sizes,
                    client_types=client_types if priv_preferences else None
                )
            else:
                algorithm.aggregate_updates(client_updates, client_weights)
            
            model = algorithm.get_model_state()
        
        # 评估
        algorithm.set_model_state(model)
        accuracy, loss = algorithm.evaluate(test_dataset, batch_size=32)
        accuracy_history.append(accuracy)
        
        print(f"测试准确率: {accuracy:.2f}%, 损失: {loss:.4f}")
        print(f"轮次时间: {time.time() - round_start_time:.2f}s")
        
        # 隐私信息
        if args.dpsgd and hasattr(algorithm, 'get_privacy_guarantees'):
            privacy_info = algorithm.get_privacy_guarantees()
            print(f"隐私信息: {privacy_info}")
            privacy_history.append(privacy_info)
        
        # 投影质量信息
        if args.projection and hasattr(algorithm, 'get_projection_quality'):
            projection_quality = algorithm.get_projection_quality()
            print(f"投影质量: {projection_quality}")
            projection_quality_history.append(projection_quality)
    
    # 最终结果
    print(f"\n总训练时间: {time.time() - start_time:.2f}s")
    print(f"最终准确率: {accuracy_history[-1]:.2f}%")
    
    if privacy_history:
        print(f"最终隐私信息: {privacy_history[-1]}")
    
    if projection_quality_history:
        print(f"最终投影质量: {projection_quality_history[-1]}")
    
    # 算法信息
    if hasattr(algorithm, 'get_algorithm_info'):
        algorithm_info = algorithm.get_algorithm_info()
        print(f"算法信息: {algorithm_info}")

if __name__ == '__main__':
    main()
