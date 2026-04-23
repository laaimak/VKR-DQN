import pytest
import torch
import numpy as np
import sys
import os

from mixer import VDNMixer
from model import DQNetwork
from memory import ReplayBuffer
from agent import get_role_by_agent_id

sys.path.append(os.path.join(os.path.dirname(__file__), "python_vdn"))


def test_vdn_mixer_math():
    """Проверка свойства IGM: аддитивная декомпозиция"""
    mixer = VDNMixer()
    # Батч из 2 состояний, 3 агента (индивидуальные Q-значения)
    q_values = torch.tensor([
        [1.5, 2.0, -0.5],
        [0.0, 1.0, 1.0]
    ])
    
    q_tot = mixer(q_values)
    
    assert q_tot.shape == (2, 1), "Размерность выхода миксера должна быть [batch, 1]"
    assert q_tot[0, 0].item() == 3.0, "Ошибка математики в VDNMixer (батч 1)"
    assert q_tot[1, 0].item() == 2.0, "Ошибка математики в VDNMixer (батч 2)"

def test_dqn_architecture():
    """Проверка размерностей тензоров в нейронной сети"""
    # 18 признаков (StateBuilder) -> 8 макро-действий (sample_player)
    net = DQNetwork(input_dim=18, output_dim=8)
    dummy_state = torch.rand(1, 18)
    output = net(dummy_state)
    
    assert output.shape == (1, 8), "Сеть должна возвращать вектор Q-значений для 8 действий"

def test_replay_buffer_smdp():
    """Проверка циклического буфера для SMDP (наличие tau и логика FIFO)"""
    buffer = ReplayBuffer()
    buffer.buffer = type(buffer.buffer)(maxlen=2) 
    
    state_mock = [0.0] * 18
    # Добавляем 3 перехода
    buffer.push(state_mock, 1, 10.0, state_mock, False, 15)
    buffer.push(state_mock, 2, 20.0, state_mock, False, 20)
    buffer.push(state_mock, 3, 30.0, state_mock, True, 5) # Вытеснит первый
    
    assert len(buffer.buffer) == 2, "Буфер превысил параметр capacity"
    
    # Проверяем структуру кортежа: (state, action, reward, next_state, done, tau)
    first_element = buffer.buffer[0]
    assert len(first_element) == 6, "Кортеж перехода должен содержать ровно 6 элементов (SMDP)"
    assert first_element[2] == 20.0, "Нарушена логика FIFO"
    assert first_element[5] == 20, "Переменная tau (длительность) не сохранилась"

def test_agent_roles():
    """Проверка привязки конфигураций и логики Reward Shaping"""
    assert get_role_by_agent_id(1) == "goalie"
    assert get_role_by_agent_id(4) == "defender"
    assert get_role_by_agent_id(8) == "midfielder"
    assert get_role_by_agent_id(10) == "forward"