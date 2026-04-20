"""
VDN Central Server — Flask HTTP сервер.

Агенты подключаются к одному серверу. Конфиг читается из CONFIG_PATH.
Если в конфиге agents.fresh_start=true — обучение с нуля.

Эндпоинты:
    POST /step   — агент сообщает своё состояние и получает действие
    POST /reset  — агент сигнализирует начало нового эпизода
    GET  /status — текущее состояние тренера (для отладки)
"""

import os
import sys
import json
import csv
import threading
import logging
from datetime import datetime
from flask import Flask, request, jsonify

from vdn_trainer import VDNTrainer

# ─── Конфигурация ────────────────────────────────────────────────────────────
# Можно передать путь к конфигу через аргумент командной строки:
#   python3 server.py config_vdn_4v4.json
_cfg_arg    = sys.argv[1] if len(sys.argv) > 1 else "config_vdn.json"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), _cfg_arg)

with open(CONFIG_PATH) as _f:
    _cfg = json.load(_f)

LOGS_PATH = _cfg.get("logs_path",
            "/Users/laaimak/Desktop/VKR/helios-qmix/src/logs")
PORT      = _cfg.get("server", {}).get("port", 6100)

# AGENT_IDS берём из конфига (секция agents.ids) или дефолт 10 агентов
AGENT_IDS   = _cfg.get("agents", {}).get("ids", [2,3,4,5,6,7,8,9,10,11])
N_AGENTS    = len(AGENT_IDS)
TRAIN_EVERY = N_AGENTS

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# ─── Глобальное состояние ────────────────────────────────────────────────────
trainer: VDNTrainer = None
lock = threading.Lock()

# Последнее известное состояние и действие каждого агента
agent_states:  dict = {}   # agent_id → state
agent_actions: dict = {}   # agent_id → action
transitions_pending = [0]  # счётчик накопленных переходов
step_count          = [0]
match_count         = [0]

# CSV-лог результатов матчей
MATCH_LOG_PATH = os.path.join(
    "/Users/laaimak/Desktop/VKR/helios-qmix/src/logs",
    "vdn_match_results.csv"
)
LOSS_LOG_PATH = os.path.join(
    "/Users/laaimak/Desktop/VKR/helios-qmix/src/logs",
    "vdn_loss.csv"
)
_match_log_initialized = [False]
_loss_log_initialized  = [False]

def _init_match_log():
    if not _match_log_initialized[0]:
        with open(MATCH_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow([
                    "match", "timestamp",
                    "left_score", "right_score",
                    "result",
                    "steps_done", "epsilon",
                    "buffer_size"
                ])
        _match_log_initialized[0] = True

def _log_loss(steps_done, epsilon, loss):
    if not _loss_log_initialized[0]:
        with open(LOSS_LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow(["steps_done", "epsilon", "loss"])
        _loss_log_initialized[0] = True
    with open(LOSS_LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([steps_done, f"{epsilon:.4f}", f"{loss:.6f}"])


@app.route("/reset", methods=["POST"])
def reset():
    """Начало нового эпизода — сбрасываем состояние агента."""
    data     = request.json
    agent_id = int(data["agent_id"])
    state    = data["state"]

    action = trainer.select_action(agent_id, state)

    with lock:
        agent_states[agent_id]  = state
        agent_actions[agent_id] = action

    return jsonify({"action": action,
                    "epsilon": trainer.epsilon,
                    "steps":   trainer.steps_done})


@app.route("/step", methods=["POST"])
def step():
    """
    Агент завершил макро-действие и сообщает:
      state  — новое состояние s'
      reward — накопленная награда за макро-действие
      done   — конец эпизода

    Сервер записывает переход в shared buffer.
    Когда накопилось TRAIN_EVERY переходов — делает train_step.
    Немедленно возвращает новое действие (без ожидания других агентов).
    """
    data     = request.json
    agent_id = int(data["agent_id"])
    state    = data["state"]
    reward   = float(data["reward"])
    done     = bool(data.get("done", False))

    with lock:
        prev_state  = agent_states.get(agent_id)
        prev_action = agent_actions.get(agent_id)

    # Если есть предыдущий шаг — записываем переход
    if prev_state is not None and prev_action is not None:
        # Собираем состояния всех агентов для совместного перехода
        with lock:
            all_states      = {aid: agent_states.get(aid,  [0.0]*18) for aid in AGENT_IDS}
            all_actions     = {aid: agent_actions.get(aid, 0)         for aid in AGENT_IDS}
            all_next_states = dict(all_states)
            all_next_states[agent_id] = state  # только этот агент обновился

        team_reward = reward  # индивидуальная награда как часть командной

        trainer.push_transition(
            states      = all_states,
            actions     = all_actions,
            team_reward = team_reward,
            next_states = all_next_states,
            done        = done,
        )

        with lock:
            transitions_pending[0] += 1
            should_train = (transitions_pending[0] >= TRAIN_EVERY)
            if should_train:
                transitions_pending[0] = 0

        if should_train:
            loss = trainer.train_step()
            step_count[0] += 1
            if loss is not None:
                _log_loss(trainer.steps_done, trainer.epsilon, loss)
            if step_count[0] % 200 == 0:
                trainer.save()
                print(f"[VDN] step={trainer.steps_done} "
                      f"eps={trainer.epsilon:.4f}"
                      + (f" loss={loss:.4f}" if loss else ""))

    # Выбираем новое действие и запоминаем
    action = trainer.select_action(agent_id, state)
    with lock:
        agent_states[agent_id]  = state
        agent_actions[agent_id] = action

    return jsonify({"action":  action,
                    "epsilon": trainer.epsilon,
                    "steps":   trainer.steps_done})


@app.route("/match_result", methods=["POST"])
def match_result():
    """
    Скрипт запуска сообщает результат матча после его завершения.
    Тело: {"left_score": int, "right_score": int}
    """
    data = request.json or {}
    left  = int(data.get("left_score",  0))
    right = int(data.get("right_score", 0))

    if left > right:
        result = "win"
    elif left < right:
        result = "loss"
    else:
        result = "draw"

    with lock:
        match_count[0] += 1
        mn = match_count[0]

    _init_match_log()
    with open(MATCH_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            mn,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            left, right,
            result,
            trainer.steps_done,
            f"{trainer.epsilon:.4f}",
            len(trainer.buffer),
        ])

    print(f"[VDN] Матч {mn}: {left}:{right} ({result}) | "
          f"steps={trainer.steps_done} eps={trainer.epsilon:.4f}")

    return jsonify({"match": mn, "result": result})


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "steps_done": trainer.steps_done,
        "epsilon":    trainer.epsilon,
        "buffer_size": len(trainer.buffer),
        "agents_ready": list(agent_states.keys()),
    })


# ─── Запуск ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(LOGS_PATH, exist_ok=True)

    fresh = _cfg.get("agents", {}).get("fresh_start", False)
    print("=" * 50)
    print(" VDN Central Server")
    print(f" Конфиг:  {os.path.basename(CONFIG_PATH)}")
    print(f" Порт:    {PORT}")
    print(f" Агенты:  {AGENT_IDS}")
    print(f" Логи:    {LOGS_PATH}")
    print(f" Режим:   {'С НУЛЯ (fresh_start)' if fresh else 'продолжение / IQL инициализация'}")
    print("=" * 50)

    trainer = VDNTrainer(CONFIG_PATH, LOGS_PATH)

    app.run(host="0.0.0.0", port=PORT, threaded=True)
