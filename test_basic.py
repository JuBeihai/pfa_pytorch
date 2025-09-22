#!/usr/bin/env python3
"""
Basic test script for PFA PyTorch implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.federated import FederatedDataSplitter
from src.models.cnn import MNISTCNN
from src.algorithms.fedavg import FedAvg
import torch

def test_basic_functionality():
    """Test basic functionality"""
    print("Testing basic functionality...")
    
    # Test model creation
    model = MNISTCNN()
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test data loading
    data_splitter = FederatedDataSplitter('mnist', num_clients=3, iid=True)
    client_datasets = data_splitter.create_clients()
    test_dataset = data_splitter.get_test_dataset()
    print(f"✓ Data loaded: {len(client_datasets)} clients, {len(test_dataset)} test samples")
    
    # Test FedAvg
    device = torch.device('cpu')
    algorithm = FedAvg(model, lr=0.1, device=device)
    
    # Test one round
    client_updates = []
    for i, dataset in enumerate(client_datasets):
        algorithm.local_update(dataset, epochs=1, batch_size=16)
        client_updates.append(algorithm.get_model_state())
    
    algorithm.aggregate_updates(client_updates)
    accuracy, loss = algorithm.evaluate(test_dataset, batch_size=16)
    print(f"✓ FedAvg test completed: Accuracy = {accuracy:.2f}%, Loss = {loss:.4f}")
    
    print("All basic tests passed! ✓")

if __name__ == '__main__':
    test_basic_functionality()
