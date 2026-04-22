"""
VDN Mixer — 11-я нейронная сеть.
"""

import torch
import torch.nn as nn


class VDNMixer(nn.Module):
    """
    VDN Mixer: Q_total = sum(Q_1, Q_2, ..., Q_n)
    """

    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs: torch.Tensor) -> torch.Tensor:
        """
        Параметры:
            agent_qs: [batch, n_agents] — индивидуальные Q-значения
                      для выбранных действий каждого агента

        Возвращает:
            q_total: [batch, 1] — командная Q-функция
        """
        return agent_qs.sum(dim=1, keepdim=True)
