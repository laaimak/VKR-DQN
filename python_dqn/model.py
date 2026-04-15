import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


class DQNetwork(nn.Module):
    """
    Аппроксиматор функции ценности Q_i(s^i, o^i; theta_i).

    Архитектура: многослойный перцептрон (MLP).
    Входной слой: вектор состояния s_t размерностью 18.
    Скрытые слои: два слоя с функцией активации ReLU: f(x) = max(0, x).
    Выходной слой: 8 нейронов — по одному на каждое макро-действие из O.
    """

    def __init__(self, config_path: str = None):
        super(DQNetwork, self).__init__()

        cfg = load_config(config_path)
        net_cfg = cfg["network"]

        input_dim  = net_cfg["input_dim"]   # 18
        hidden_dim = net_cfg["hidden_dim"]  # 128
        output_dim = net_cfg["output_dim"]  # 8

        # Информационный уровень: 18 признаков состояния
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Выходной слой: 8 макро-действий
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Инициализация выходного слоя малыми весами
        # Это предотвращает взрывной рост Q-значений в начале обучения
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямое распространение.
        x: вектор состояния s_t размерностью [batch, 18]
        Возвращает: Q-значения для каждого макро-действия [batch, 8]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)