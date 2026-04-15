import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import os
import json
from model import DQNetwork
from memory import ReplayBuffer


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


class SMDPAgent:
    """
    DQN агент для Semi-Markov Decision Process.

    Каждый агент обучается независимо (Independent Q-Learning):
    своя нейросеть, свой буфер воспроизведения, свои веса.
    """
    def __init__(self, agent_id: int, config_path: str = None):
        
        forced_id = os.environ.get("AGENT_FORCE_ID")
        if forced_id is not None:
            agent_id = int(forced_id)
        
        self.agent_id = agent_id
        self.config_path = config_path
        cfg = load_config(config_path)

        learn_cfg = cfg["learning"]
        net_cfg   = cfg["network"]
        train_cfg = cfg["training"]

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Apple M2
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.action_dim = net_cfg["output_dim"]
        self.gamma      = learn_cfg["gamma"]

        # ε-жадная стратегия: ε(t) = ε_min + (ε_start - ε_min) * e^{-λt}
        self.epsilon       = learn_cfg["epsilon_start"]
        self.epsilon_start = learn_cfg["epsilon_start"]
        self.epsilon_min   = learn_cfg["epsilon_min"]
        self.epsilon_decay = learn_cfg["epsilon_decay"]
        self.steps_done    = 0

        # Основная сеть обновляется каждый шаг,
        # целевая — каждые target_update_freq шагов
        self.policy_net = DQNetwork(config_path).to(self.device)
        self.target_net = DQNetwork(config_path).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learn_cfg["learning_rate"]
        )

        self.memory = ReplayBuffer(config_path)

        self.batch_size         = learn_cfg["batch_size"]
        self.target_update_freq = learn_cfg["target_update_freq"]
        self.log_interval       = train_cfg["log_interval"]
        self.save_interval      = train_cfg["save_interval"]
        self.elite_weights_path = train_cfg["elite_weights_path"]
        self.logs_path          = train_cfg["logs_path"]

        self.episode_reward = 0.0

        os.makedirs(self.logs_path, exist_ok=True)

        if self.agent_id == 1:
            self.role = "goalie"
        elif 2 <= self.agent_id <= 5:
            self.role = "defender"
        elif 6 <= self.agent_id <= 8:
            self.role = "midfielder"
        else:
            self.role = "forward"

        # Формируем пути динамически
        self.log_file = os.path.join(
            self.logs_path,
            f"agent_{self.agent_id}_steps.csv"
        )

        self.episode_log_file = os.path.join(
            self.logs_path,
            f"agent_{self.agent_id}_episode_rewards.csv"
        )
        
        # Личный чекпоинт конкретного номера (чтобы продолжить после паузы)
        agent_checkpoint = os.path.join(
            self.logs_path, f"agent_{self.agent_id}_checkpoint.pth"
        )
        
        # "Мастер-веса" для всей роли (например, лучшие знания всех защитников)
        self.role_master_path = os.path.join(
            self.logs_path, f"master_{self.role}.pth"
        )

        # 1. Свой чекпоинт (самый точный прогресс конкретного игрока)
        if os.path.exists(agent_checkpoint):
            self.load_weights(agent_checkpoint)
            print(f"[Agent {self.agent_id}] Загружен личный чекпоинт.")
            
        # 2. Мастер-веса роли (если своего нет, берем опыт "старших братьев" по позиции)
        elif os.path.exists(self.role_master_path):
            self.load_weights(self.role_master_path)
            print(f"[Agent {self.agent_id}] Наследуем мастер-веса роли: {self.role}")
            
        # 3. Общие элитные веса (если они прописаны в конфиге как база для всех)
        elif self.elite_weights_path and os.path.exists(self.elite_weights_path):
            self.load_weights(self.elite_weights_path)
            print(f"[Agent {self.agent_id}] Используем общие элитные веса.")
            
        else:
            print(f"[Agent {self.agent_id}] Начинаем обучение с нуля для роли {self.role}.")

        
        
    def act(self, state: list) -> int:
        """
        ε-жадный выбор макро-действия.
        Возвращает action_id от 1 до 8.
        """
        if random.random() < self.epsilon:
            return random.randint(1, self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values).item()) + 1

    def train(self) -> float | None:
        """
        Один шаг оптимизации по уравнению Беллмана для SMDP:

            y_t = R_t + gamma^tau * max_a Q(s_{t+tau}, a; theta^-)
            L   = MSE(Q(s_t, a_t; theta), y_t)
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, taus = \
            self.memory.sample(self.batch_size)

        states      = states.to(self.device)
        actions     = actions.to(self.device).long()
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)
        taus        = taus.to(self.device)

        current_q = self.policy_net(states).gather(
            1, (actions - 1).view(-1, 1)
        )

        with torch.no_grad():
            next_q   = self.target_net(next_states).max(1)[0]
            discount = torch.pow(self.gamma, taus)
            target_q = rewards + (discount * next_q * (1 - dones))

        # Huber loss устойчивее MSE при редких больших TD-ошибках.
        loss = F.smooth_l1_loss(current_q, target_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1
        self._update_epsilon()
        
        if self.steps_done % 100 == 0:
            self._check_and_save_record()

        if self.steps_done % self.log_interval == 0:
            self._log_step(loss.item(), rewards.mean().item())

        if self.steps_done % self.save_interval == 0:
            checkpoint_path = os.path.join(
                self.logs_path, f"agent_{self.agent_id}_checkpoint.pth"
            )
            self.save_weights(checkpoint_path)

        # Обновляем целевую сеть
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def push_transition(self,
                        state: list,
                        action: int,
                        reward: float,
                        next_state: list,
                        done: bool,
                        tau: int):
        """Добавляет переход в буфер воспроизведения."""
        # Клиппинг награды стабилизирует диапазон target_q.
        reward_for_training = max(-10.0, min(10.0, reward))
        self.episode_reward += reward
        self.memory.push(state, action, reward_for_training, next_state, done, tau)

    def _update_epsilon(self):
        """Экспоненциальное убывание ε по мере накопления опыта."""
        self.epsilon = self.epsilon_min + (
            self.epsilon_start - self.epsilon_min
        ) * math.exp(-self.epsilon_decay * self.steps_done)
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def save_weights(self, path: str):
        torch.save({
            'policy_net':    self.policy_net.state_dict(),
            'target_net':    self.target_net.state_dict(),
            'optimizer':     self.optimizer.state_dict(),
            'steps_done':    self.steps_done,
            'epsilon':       self.epsilon,
            'epsilon_start': self.epsilon_start,
        }, path)

    def load_weights(self, path: str):
        """Загружает веса, состояние оптимизатора, шаги и Эпсилон."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            
            # Загружаем оптимизатор, чтобы Adam помнил моментум
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            # Восстанавливаем прогресс обучения
            self.steps_done    = checkpoint.get('steps_done', 0)
            self.epsilon       = checkpoint.get('epsilon', self.epsilon_start)
            self.epsilon_start = checkpoint.get('epsilon_start', self.epsilon_start)

            print(f"[Agent {self.agent_id}] 🧠 Загружен чекпоинт из {os.path.basename(path)}.")
            print(f"[Agent {self.agent_id}] 🔄 Продолжаем! Шаги: {self.steps_done}, Эпсилон: {self.epsilon:.4f}")
            
    def save_elite_if_best(self, all_rewards: dict):
        """Индивидуальное сохранение весов: бьем собственный исторический рекорд."""
        # Берем награду из словаря (там только мы сами, так как C++ шлет только себя)
        match_reward = all_rewards.get(self.agent_id, -float('inf'))
        
        # Файлы для конкретного агента
        record_file = os.path.join(self.logs_path, f"record_agent_{self.agent_id}.json")
        personal_weights_path = os.path.join(self.logs_path, f"best_weights_agent_{self.agent_id}.pth")
        
        # Читаем старый рекорд
        personal_best = -float('inf')
        if os.path.exists(record_file):
            try:
                with open(record_file, "r") as f:
                    personal_best = json.load(f).get("best_reward", -float('inf'))
            except json.JSONDecodeError:
                pass
        
        # Сравниваем
        if match_reward > personal_best:
            with open(record_file, "w") as f:
                json.dump({"best_reward": match_reward, "steps_done": self.steps_done}, f)
            
            self.save_weights(personal_weights_path)
            
            self.save_weights(self.role_master_path) 
            print(f"[Agent {self.agent_id}] Мастер-веса роли {self.role} обновлены!")
        else:
            print(f"[Agent {self.agent_id}] Личный рекорд не побит. "
                  f"(Текущий: {match_reward:.2f}, Рекорд: {personal_best:.2f})")

    def reset_episode(self):
        """Сбрасывает накопленную награду в начале нового матча."""
        self.episode_reward = 0.0

    def log_episode_result(self, our_score: int, opp_score: int, episode_reward: float | None = None):
        """Пишет итог матча в отдельный CSV: награда, счет, шаги, epsilon."""
        if episode_reward is None:
            episode_reward = self.episode_reward

        file_has_content = os.path.isfile(self.episode_log_file) and os.path.getsize(self.episode_log_file) > 0

        match_id = 1
        if file_has_content:
            with open(self.episode_log_file, mode='r') as f:
                # -1 для заголовка
                line_count = sum(1 for _ in f)
                match_id = max(1, line_count)

        with open(self.episode_log_file, mode='a') as f:
            if not file_has_content:
                f.write("Match,EpisodeReward,ScoreFor,ScoreAgainst,StepsDone,Epsilon\n")
            f.write(
                f"{match_id},{episode_reward:.6f},{our_score},{opp_score},{self.steps_done},{self.epsilon:.4f}\n"
            )
            f.flush()
            os.fsync(f.fileno())
    
    def _check_and_save_record(self):
        """Страховка: сохраняем рекорд ПРЯМО ВО ВРЕМЯ МАТЧА."""
        record_file = os.path.join(self.logs_path, f"record_agent_{self.agent_id}.json")
        personal_weights_path = os.path.join(self.logs_path, f"best_weights_agent_{self.agent_id}.pth")

        personal_best = -float('inf')
        if os.path.exists(record_file):
            try:
                with open(record_file, "r") as f:
                    personal_best = json.load(f).get("best_reward", -float('inf'))
            except json.JSONDecodeError:
                pass

        # Если текущая накопленная награда УЖЕ больше рекорда — сохраняем!
        if self.episode_reward > personal_best:
            with open(record_file, "w") as f:
                json.dump({"best_reward": self.episode_reward, "steps_done": self.steps_done}, f)
            
            self.save_weights(personal_weights_path)
            self.save_weights(self.role_master_path)
            print(f"[Agent {self.agent_id}] 🔥 СТРАХОВКА СРАБОТАЛА: Награда {self.episode_reward:.2f} побила рекорд! Мастер-веса обновлены.")

    
    def _log_step(self, loss: float, avg_reward: float):
        file_has_content = os.path.isfile(self.log_file) and os.path.getsize(self.log_file) > 0
        with open(self.log_file, mode='a') as f:
            if not file_has_content:
                f.write("Step,Loss,AvgReward,Epsilon\n")
            f.write(
                f"{self.steps_done},{loss:.6f},"
                f"{avg_reward:.6f},{self.epsilon:.4f}\n"
            )
            f.flush()
            os.fsync(f.fileno())