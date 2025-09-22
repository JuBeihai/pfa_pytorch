import torch
import math
from typing import Tuple, Optional

class PrivacyAccountant:
    """
    差分隐私会计器，完全匹配 TensorFlow 版本的实现
    使用简化的噪声计算公式
    """
    
    def __init__(self, epsilon: float, delta: float, noise_multiplier: Optional[float] = None):
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.accum_bgts = 0
        self.tmp_accum_bgts = 0
        self.finished = False
        self.curr_steps = 0
        
    def precheck(self, dataset_size: int, batch_size: int, loc_steps: int) -> bool:
        """
        预检查客户端是否可以参与下一轮训练
        匹配 TensorFlow 版本的 precheck 逻辑
        """
        if self.finished:
            return False
        
        # 计算临时累积预算
        tmp_steps = self.curr_steps + loc_steps
        q = batch_size * 1.0 / dataset_size
        tmp_accum_bgts = 10 * q * math.sqrt(tmp_steps * (-math.log10(self.delta))) / self.noise_multiplier
        
        # 如果预算耗尽，设置状态为完成
        if self.epsilon - tmp_accum_bgts < 0:
            self.finished = True
            return False
        else:
            self.tmp_accum_bgts = tmp_accum_bgts
            return True
    
    def update(self, loc_steps: int) -> float:
        """
        更新隐私预算消耗
        匹配 TensorFlow 版本的 update 逻辑
        """
        self.curr_steps += loc_steps
        self.accum_bgts = self.tmp_accum_bgts
        self.tmp_accum_bgts = 0
        return self.accum_bgts
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        获取已消耗的隐私预算
        """
        return self.accum_bgts, self.delta
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """
        获取剩余的隐私预算
        """
        remaining_epsilon = max(0, self.epsilon - self.accum_bgts)
        return remaining_epsilon, self.delta
    
    def is_budget_exhausted(self) -> bool:
        """
        检查隐私预算是否耗尽
        """
        return self.finished
    
    def compute_noise_multiplier(self, N: int, L: int, T: int, epsilon: float, delta: float) -> float:
        """
        计算噪声乘数，匹配 TensorFlow 版本的公式
        """
        q = L / N
        nm = 10 * q * math.sqrt(T * (-math.log10(delta))) / epsilon
        return nm