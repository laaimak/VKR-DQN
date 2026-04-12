import random
from collections import deque
import numpy as np
import torch
from typing import Tuple
import json
import os


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


class ReplayBuffer:
    """
    Циклический буфер для хранения переходов (s, a, r, s', done, tau).
    При заполнении вытесняет самые старые записи.
    """

    def __init__(self, config_path: str = None):
        cfg = load_config(config_path)
        capacity = cfg["learning"]["buffer_size"]
        self.buffer = deque(maxlen=capacity)

    def push(self,
             state: list,
             action: int,
             reward: float,
             next_state: list,
             done: bool,
             tau: int):
        """Добавляет переход в буфер."""
        self.buffer.append((state, action, reward, next_state, done, tau))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Случайная выборка мини-батча — разрывает корреляции между шагами."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, tau = map(np.stack, zip(*batch))

        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done),
            torch.FloatTensor(tau)
        )

    def __len__(self) -> int:
        return len(self.buffer)