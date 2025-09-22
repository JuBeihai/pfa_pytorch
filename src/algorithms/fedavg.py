import torch
import copy
from typing import List, Dict
from torch.utils.data import DataLoader

class FedAvg:
    def __init__(self, model, lr=0.1, device='cpu'):
        self.model = model
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def local_update(self, dataset, local_steps=100, batch_size=4):
        """
        匹配 TensorFlow 版本的本地更新逻辑
        使用 local_steps 而不是 epochs，每步随机采样一个 batch
        """
        self.model.train()
        
        # 获取数据集大小
        dataset_size = len(dataset)
        
        for step in range(local_steps):
            # 随机采样一个 batch，匹配 TensorFlow 版本
            batch_indices = torch.randperm(dataset_size)[:batch_size]
            
            # 获取 batch 数据
            batch_data = []
            batch_targets = []
            for idx in batch_indices:
                data, target = dataset[idx]
                batch_data.append(data)
                batch_targets.append(target)
            
            data = torch.stack(batch_data).to(self.device)
            target = torch.stack(batch_targets).to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
    
    def get_model_state(self):
        """获取模型状态"""
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_state(self, state_dict):
        """设置模型状态"""
        self.model.load_state_dict(state_dict)
    
    def aggregate_updates(self, client_updates, client_weights=None):
        """服务器聚合客户端更新"""
        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # 加权平均
        aggregated_state = {}
        for key in client_updates[0].keys():
            aggregated_state[key] = sum(
                client_updates[i][key] * client_weights[i] 
                for i in range(len(client_updates))
            )
        
        self.set_model_state(aggregated_state)
    
    def evaluate(self, test_dataset, batch_size=32):
        """评估模型性能"""
        self.model.eval()
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        correct = 0
        total = 0
        total_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss