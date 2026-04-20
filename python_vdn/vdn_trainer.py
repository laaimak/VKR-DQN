"""
VDN Trainer — центральный тренер для всех агентов.

Архитектура:
  - N индивидуальных DQN-агентов (задаётся через конфиг)
  - 1 VDNMixer — суммирует Q-значения → Q_total
  - Shared replay buffer — хранит совместные переходы всей команды

Протокол обучения (один шаг):
  1. Каждый агент выбирает действие по своей Q-сети (ε-greedy)
  2. Все агенты выполняют действия в симуляторе
  3. Тренер собирает (s_i, a_i, r_team, s'_i) от всех агентов
  4. Пишет совместный переход в shared buffer
  5. Сэмплирует батч → считает Q_total через VDNMixer
  6. Loss = (Q_total - (r_team + γ * max Q_total_next))²
  7. Обновляет все N сетей + mixer одним backward pass
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from typing import Dict, List, Optional

from model import DQN
from mixer import VDNMixer


class VDNSharedBuffer:
    """
    Совместный replay buffer для всей команды.

    Каждый переход содержит состояния и действия ВСЕХ агентов:
        (states, actions, team_reward, next_states, done)

    states:      [n_agents, state_dim]
    actions:     [n_agents]
    team_reward: float (общая командная награда)
    next_states: [n_agents, state_dim]
    done:        bool
    """

    def __init__(self, capacity: int, n_agents: int, state_dim: int):
        self.capacity   = capacity
        self.n_agents   = n_agents
        self.state_dim  = state_dim
        self.buffer: List[dict] = []
        self.pos = 0

    def push(self, states, actions, team_reward, next_states, done):
        transition = {
            "states":      np.array(states,      dtype=np.float32),
            "actions":     np.array(actions,     dtype=np.int64),
            "reward":      float(team_reward),
            "next_states": np.array(next_states, dtype=np.float32),
            "done":        float(done),
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states      = torch.FloatTensor(np.stack([b["states"]      for b in batch]))
        actions     = torch.LongTensor(np.stack([b["actions"]      for b in batch]))
        rewards     = torch.FloatTensor([b["reward"]               for b in batch]).unsqueeze(1)
        next_states = torch.FloatTensor(np.stack([b["next_states"] for b in batch]))
        dones       = torch.FloatTensor([b["done"]                 for b in batch]).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class VDNTrainer:
    """
    Центральный VDN тренер.

    AGENT_IDS и роли читаются из конфига (секция "agents").
    Если fresh_start=true — обучение с нуля, без загрузки IQL весов.
    """

    STATE_DIM = 18
    N_ACTIONS = 8

    def __init__(self, config_path: str, logs_path: str):
        self.logs_path = logs_path
        os.makedirs(logs_path, exist_ok=True)

        with open(config_path) as f:
            cfg = json.load(f)

        learn = cfg["learning"]
        self.gamma              = learn["gamma"]
        self.batch_size         = learn["batch_size"]
        self.target_update_freq = learn["target_update_freq"]
        self.lr                 = learn["learning_rate"]
        self.epsilon_start      = learn["epsilon_start"]
        self.epsilon_min        = learn["epsilon_min"]
        self.epsilon_decay      = learn["epsilon_decay"]
        self.buffer_size        = learn["buffer_size"]

        # Читаем список агентов из конфига (или дефолт 10 агентов)
        agents_cfg = cfg.get("agents", {})
        self.AGENT_IDS: List[int] = agents_cfg.get(
            "ids", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        )
        raw_roles = agents_cfg.get("roles", {})
        # Роли: {agent_id(int) → "defender"/"midfielder"/"forward"}
        self.AGENT_ROLES: Dict[int, str] = {
            int(k): v for k, v in raw_roles.items()
        } if raw_roles else {
            2: "defender",  3: "defender",  4: "defender",  5: "defender",
            6: "midfielder",7: "midfielder", 8: "midfielder", 9: "midfielder",
            10: "forward",  11: "forward",
        }
        self.fresh_start: bool = agents_cfg.get("fresh_start", False)
        self.N_AGENTS = len(self.AGENT_IDS)

        # Откуда грузить веса (по умолчанию — та же папка что и logs_path)
        self.weights_load_path: str = cfg.get("weights_load_path", logs_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Индивидуальные Q-сети (policy + target) для каждого агента
        self.policy_nets: Dict[int, DQN] = {}
        self.target_nets: Dict[int, DQN] = {}
        for aid in self.AGENT_IDS:
            self.policy_nets[aid] = DQN(self.STATE_DIM, self.N_ACTIONS).to(self.device)
            self.target_nets[aid] = DQN(self.STATE_DIM, self.N_ACTIONS).to(self.device)
            self.target_nets[aid].load_state_dict(self.policy_nets[aid].state_dict())
            self.target_nets[aid].eval()

        # VDN Mixer — суммирует Q_i → Q_total
        self.mixer        = VDNMixer().to(self.device)
        self.target_mixer = VDNMixer().to(self.device)

        # Единый оптимизатор для всех сетей + mixer
        all_params = list(self.mixer.parameters())
        for net in self.policy_nets.values():
            all_params += list(net.parameters())
        self.optimizer = optim.Adam(all_params, lr=self.lr)

        # Shared replay buffer
        self.buffer = VDNSharedBuffer(
            capacity=self.buffer_size,
            n_agents=self.N_AGENTS,
            state_dim=self.STATE_DIM,
        )

        self.steps_done = 0
        self.epsilon    = self.epsilon_start

        if self.fresh_start:
            print(f"[VDN] fresh_start=true — обучение с нуля (случайная инициализация)")
        else:
            # Пробуем загрузить VDN checkpoint (продолжение) или IQL веса (инициализация)
            if not self._try_load_vdn_checkpoint():
                self._load_pretrained_weights()

    def _try_load_vdn_checkpoint(self) -> bool:
        """Пробует загрузить VDN checkpoint из weights_load_path. Возвращает True если удалось."""
        all_found = all(
            os.path.exists(os.path.join(self.weights_load_path, f"vdn_agent_{aid}.pth"))
            for aid in self.AGENT_IDS
        )
        if all_found:
            self.load()
            return True
        return False

    def _load_pretrained_weights(self):
        """Загружает IQL веса как начальную инициализацию (не fresh_start)."""
        for aid in self.AGENT_IDS:
            role = self.AGENT_ROLES.get(aid, "defender")
            ckpt   = os.path.join(self.logs_path, f"agent_{aid}_checkpoint.pth")
            master = os.path.join(self.logs_path, f"master_{role}.pth")

            path = None
            if os.path.exists(ckpt):
                path = ckpt
            elif os.path.exists(master):
                path = master

            if path:
                try:
                    cp = torch.load(path, map_location=self.device, weights_only=False)
                    self.policy_nets[aid].load_state_dict(cp["policy_net"])
                    self.target_nets[aid].load_state_dict(cp["target_net"])
                    print(f"[VDN] Agent #{aid}: загружены IQL веса из {os.path.basename(path)}")
                except Exception as e:
                    print(f"[VDN] Agent #{aid}: не удалось загрузить веса: {e}")
            else:
                print(f"[VDN] Agent #{aid}: случайная инициализация")

        print(f"[VDN] Старт с IQL весов (VDN обучение с нуля)")

    def select_action(self, agent_id: int, state: List[float]) -> int:
        """ε-greedy выбор действия для агента."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_nets[agent_id](state_t).argmax(dim=1).item()

    def push_transition(self, states: dict, actions: dict,
                        team_reward: float, next_states: dict, done: bool):
        """
        Записывает совместный переход в shared buffer.
        states, actions, next_states: {agent_id: data}
        """
        s  = [states[aid]      for aid in self.AGENT_IDS]
        a  = [actions[aid]     for aid in self.AGENT_IDS]
        ns = [next_states[aid] for aid in self.AGENT_IDS]
        self.buffer.push(s, a, team_reward, ns, done)

    def train_step(self) -> Optional[float]:
        """Один шаг обучения VDN."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Q-значения для выбранных действий
        agent_qs = []
        for i, aid in enumerate(self.AGENT_IDS):
            q_vals = self.policy_nets[aid](states[:, i, :])
            q_a    = q_vals.gather(1, actions[:, i].unsqueeze(1))
            agent_qs.append(q_a)
        agent_qs = torch.cat(agent_qs, dim=1)   # [batch, n_agents]
        q_total  = self.mixer(agent_qs)          # [batch, 1]

        # Target Q-total
        with torch.no_grad():
            target_agent_qs = []
            for i, aid in enumerate(self.AGENT_IDS):
                q_next = self.target_nets[aid](next_states[:, i, :]).max(1)[0].unsqueeze(1)
                target_agent_qs.append(q_next)
            target_agent_qs = torch.cat(target_agent_qs, dim=1)
            q_total_next    = self.target_mixer(target_agent_qs)
            q_target        = rewards + self.gamma * q_total_next * (1 - dones)

        # Loss и обновление
        loss = nn.SmoothL1Loss()(q_total, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        for net in self.policy_nets.values():
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1
        self._update_epsilon()

        if self.steps_done % self.target_update_freq == 0:
            for aid in self.AGENT_IDS:
                self.target_nets[aid].load_state_dict(self.policy_nets[aid].state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        return loss.item()

    def _update_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) \
                       * np.exp(-self.epsilon_decay * self.steps_done)

    def save(self):
        """Сохраняет все сети + mixer."""
        for aid in self.AGENT_IDS:
            path = os.path.join(self.logs_path, f"vdn_agent_{aid}.pth")
            torch.save({
                "policy_net": self.policy_nets[aid].state_dict(),
                "target_net": self.target_nets[aid].state_dict(),
                "steps_done": self.steps_done,
                "epsilon":    self.epsilon,
            }, path)
        torch.save(self.mixer.state_dict(),
                   os.path.join(self.logs_path, "vdn_mixer.pth"))
        print(f"[VDN] Сохранено. steps={self.steps_done} eps={self.epsilon:.4f}")

    def load(self):
        """Загружает VDN checkpoint из weights_load_path."""
        for aid in self.AGENT_IDS:
            path = os.path.join(self.weights_load_path, f"vdn_agent_{aid}.pth")
            if os.path.exists(path):
                cp = torch.load(path, map_location=self.device, weights_only=False)
                self.policy_nets[aid].load_state_dict(cp["policy_net"])
                self.target_nets[aid].load_state_dict(cp["target_net"])
                self.steps_done = cp.get("steps_done", 0)
                self.epsilon    = cp.get("epsilon", self.epsilon_start)

        mixer_path = os.path.join(self.weights_load_path, "vdn_mixer.pth")
        if os.path.exists(mixer_path):
            self.mixer.load_state_dict(
                torch.load(mixer_path, map_location=self.device, weights_only=False)
            )
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        print(f"[VDN] Загружено из {self.weights_load_path}")
        print(f"[VDN] steps={self.steps_done} eps={self.epsilon:.4f}")
