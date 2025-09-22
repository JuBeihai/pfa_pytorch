import argparse
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.federated import FederatedDataSplitter
from src.models.cnn import MNISTCNN
from src.algorithms.fedavg import FedAvg
from src.algorithms.pfa import PFA
from src.algorithms.dp_pfa import DPPFA
from src.privacy.noise import DPFedAvg
from src.privacy.accountant import PrivacyAccountant

def main():
    parser = argparse.ArgumentParser(description='PyTorch Federated Learning with PFA')
    parser.add_argument('--algorithm', choices=['fedavg', 'pfa', 'pfa_plus', 'dp_fedavg', 'dp_pfa'], 
                       default='fedavg', help='Federated learning algorithm')
    parser.add_argument('--dataset', default='mnist', help='Dataset name')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--rounds', type=int, default=100, help='Communication rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='Local training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--dpsgd', action='store_true', help='Enable differential privacy')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Privacy budget epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='Privacy parameter delta')
    parser.add_argument('--noise_multiplier', type=float, default=None, help='Noise multiplier for DP')
    parser.add_argument('--l2_norm_clip', type=float, default=1.0, help='L2 norm clipping bound')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='Sampling rate for DP')
    parser.add_argument('--proj_dims', type=int, default=10, help='Projection dimensions for PFA')
    parser.add_argument('--iid', action='store_true', help='Use IID data distribution')
    parser.add_argument('--data_dir', default='./data', help='Data directory')
    
    args = parser.parse_args()
    
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MNISTCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 数据准备
    data_splitter = FederatedDataSplitter(
        args.dataset, args.num_clients, iid=args.iid, data_dir=args.data_dir
    )
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    
    print(f"Created {len(client_datasets)} clients")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 算法选择
    if args.algorithm == 'fedavg':
        algorithm = FedAvg(model, lr=args.lr, device=device)
    elif args.algorithm == 'pfa':
        algorithm = PFA(model, lr=args.lr, proj_dims=args.proj_dims, device=device)
    elif args.algorithm == 'dp_fedavg':
        algorithm = DPFedAvg(model, lr=args.lr, device=device,
                           epsilon=args.epsilon, delta=args.delta,
                           noise_multiplier=args.noise_multiplier,
                           l2_norm_clip=args.l2_norm_clip,
                           sample_rate=args.sample_rate)
    elif args.algorithm == 'dp_pfa':
        algorithm = DPPFA(model, lr=args.lr, proj_dims=args.proj_dims, device=device,
                        epsilon=args.epsilon, delta=args.delta,
                        noise_multiplier=args.noise_multiplier,
                        l2_norm_clip=args.l2_norm_clip,
                        sample_rate=args.sample_rate)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")
    
    print(f"Using {args.algorithm} algorithm")
    
    # 训练循环
    for round_idx in range(args.rounds):
        print(f"\n=== Round {round_idx + 1}/{args.rounds} ===")
        
        # 客户端训练
        client_updates = []
        for client_id, dataset in enumerate(client_datasets):
            # 复制全局模型到客户端
            algorithm.set_model_state(algorithm.get_model_state())
            
            # 本地训练
            algorithm.local_update(dataset, args.local_epochs, args.batch_size)
            
            # 获取更新
            client_updates.append(algorithm.get_model_state())
        
        # 服务器聚合
        algorithm.aggregate_updates(client_updates)
        
        # 评估
        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            accuracy, loss = algorithm.evaluate(test_dataset, args.batch_size)
            print(f"Round {round_idx + 1}: Test Accuracy = {accuracy:.2f}%, Test Loss = {loss:.4f}")
            
            # 显示隐私信息（如果使用差分隐私）
            if hasattr(algorithm, 'get_privacy_info'):
                privacy_info = algorithm.get_privacy_info()
                print(f"  Privacy: ε_spent = {privacy_info['epsilon_spent']:.4f}, "
                      f"ε_remaining = {privacy_info['remaining_epsilon']:.4f}, "
                      f"Compression = {privacy_info['compression_ratio']:.2f}x")
                
                if algorithm.is_budget_exhausted():
                    print("  ⚠️  Privacy budget exhausted! Stopping training.")
                    break
    
    print("\nTraining completed!")
    
    # 最终隐私报告
    if hasattr(algorithm, 'get_privacy_info'):
        privacy_info = algorithm.get_privacy_info()
        print(f"\nFinal Privacy Report:")
        print(f"  Total ε spent: {privacy_info['epsilon_spent']:.4f}")
        print(f"  Total δ spent: {privacy_info['delta_spent']:.4f}")
        print(f"  Noise multiplier: {privacy_info['noise_multiplier']:.4f}")
        print(f"  L2 norm clip: {privacy_info['l2_norm_clip']:.4f}")
        if 'compression_ratio' in privacy_info:
            print(f"  Compression ratio: {privacy_info['compression_ratio']:.2f}x")

if __name__ == '__main__':
    main()