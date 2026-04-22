#!/bin/bash

# ============================================================
# VDN Training Script
# Запускает Flask VDN-сервер (порт 6100) + 11 агентов нашей
# команды (вратарь FSM + 10 VDN полевых) против StarterAgent2D-V2.
#
# Использование:
#   ./train_vdn.sh [кол-во матчей]
# Пример:
#   ./train_vdn.sh 200
# ============================================================

MATCHES=${1:-200}

# ─── Пути ────────────────────────────────────────────────────
VDN_SERVER_DIR="/Users/laaimak/Desktop/VKR/python_vdn"
VDN_BUILD="/Users/laaimak/Desktop/VKR/helios-qmix/build"
VDN_PLAYER="${VDN_BUILD}/sample_player"

STARTER_DIR="/Users/laaimak/Desktop/VKR/StarterAgent2D-V2/build/bin"
STARTER_PLAYER="${STARTER_DIR}/sample_player"
STARTER_PLAYER_CONF="${STARTER_DIR}/player.conf"
STARTER_CONFIG_DIR="${STARTER_DIR}/formations-dt"

RCSSSERVER_DIR="/Users/laaimak/Desktop/VKR/rcssserver-master/build"
VENV="/Users/laaimak/Desktop/VKR/.venv314/bin/activate"
LOGS_DIR="/Users/laaimak/Desktop/VKR/helios-qmix/src/logs"

MATCH_DURATION=300   # half_time в секундах: 2 тайма × 300с × 10Гц = 6000 тактов

mkdir -p "$LOGS_DIR"

echo "============================================================"
echo " VDN Training: $MATCHES матчей против StarterAgent2D-V2"
echo " Flask VDN-сервер: port 6100"
echo " rcssserver: port 6000"
echo "============================================================"

# ─── Активируем venv и запускаем Flask VDN сервер ──────────
source "$VENV"
echo "[VDN] Запуск Flask сервера (порт 6100)..."
cd "$VDN_SERVER_DIR"
python3 server.py &
FLASK_PID=$!
echo "[VDN] Flask PID=$FLASK_PID"
sleep 4   # ждём загрузки весов и инициализации PyTorch

# ─── Основной цикл матчей ────────────────────────────────────
for (( i=1; i<=MATCHES; i++ )); do
    echo ""
    echo "=== Матч $i / $MATCHES ==="

    # Убиваем остатки прошлого матча
    pkill -9 -f "rcssserver" 2>/dev/null
    pkill -9 -f "sample_player" 2>/dev/null
    sleep 1

    # Запускаем rcssserver
    cd "$RCSSSERVER_DIR"
    ./rcssserver \
        server::auto_mode=true \
        server::synch_mode=true \
        server::nr_normal_halfs=2 \
        server::nr_extra_halfs=0 \
        server::penalty_shoot_outs=false \
        server::half_time=${MATCH_DURATION} \
        &
    RCSS_PID=$!
    sleep 2

    # ─── VDN команда ────────────────────────────────────────
    # Запускаем из build/ — там лежат formations-dt и player.conf
    # Запускаем из build/ — player.conf и formations-dt находятся автоматически
    cd "$VDN_BUILD"

    # Вратарь (стандартный FSM, -g)
    ./sample_player --host localhost --port 6000 -t VDN_Team -g &
    sleep 0.3

    # Полевые игроки 2-11 — VDN агенты
    for unum in 2 3 4 5 6 7 8 9 10 11; do
        ./sample_player --host localhost --port 6000 -t VDN_Team &
        sleep 0.2
    done

    # ─── Противник: StarterAgent2D-V2 ───────────────────────
    OPP_OPT="--player-config ${STARTER_PLAYER_CONF} --config_dir ${STARTER_CONFIG_DIR}"

    "$STARTER_PLAYER" $OPP_OPT --host localhost --port 6000 -t Starter -n 1 -g &
    sleep 0.3
    for unum in 2 3 4 5 6 7 8 9 10 11; do
        "$STARTER_PLAYER" $OPP_OPT --host localhost --port 6000 -t Starter &
        sleep 0.15
    done

    echo "[VDN] Матч $i запущен, ждём завершения агентов..."

    # Ждём пока VDN агенты завершатся (самовыключаются после матча)
    WAITED=0
    MAX_WAIT=$(( MATCH_DURATION / 5 + 120 ))
    while ps aux | grep -q "[s]ample_player.*VDN_Team" && [ $WAITED -lt $MAX_WAIT ]; do
        sleep 5
        WAITED=$(( WAITED + 5 ))
    done

    # Читаем счёт из episode_rewards агента 2 (последняя строка)
    LEFT_SCORE=0
    RIGHT_SCORE=0
    REWARDS_CSV="${LOGS_DIR}/vdn_agent_2_episode_rewards.csv"
    if [ -f "$REWARDS_CSV" ]; then
        LAST_LINE=$(tail -1 "$REWARDS_CSV")
        LEFT_SCORE=$(echo "$LAST_LINE"  | awk -F',' '{print $3}')
        RIGHT_SCORE=$(echo "$LAST_LINE" | awk -F',' '{print $4}')
        LEFT_SCORE=${LEFT_SCORE:-0}
        RIGHT_SCORE=${RIGHT_SCORE:-0}
    fi
    echo "[VDN] Счёт: VDN_Team ${LEFT_SCORE} : ${RIGHT_SCORE} Starter"
    curl -s -X POST http://localhost:6100/match_result \
        -H "Content-Type: application/json" \
        -d "{\"left_score\": ${LEFT_SCORE}, \"right_score\": ${RIGHT_SCORE}}" \
        > /dev/null
    echo "[VDN] Матч $i завершён (waited=${WAITED}s)"

    # Зачищаем процессы (TERM сначала — агенты успевают finalizeEpisode)
    pkill -TERM -f "sample_player" 2>/dev/null
    sleep 3
    pkill -9 -f "rcssserver" 2>/dev/null
    pkill -9 -f "sample_player" 2>/dev/null
    sleep 2
done

echo ""
echo "=== VDN обучение завершено ($MATCHES матчей) ==="
echo "[VDN] Останавливаем Flask сервер (PID=$FLASK_PID)..."
kill $FLASK_PID 2>/dev/null
wait $FLASK_PID 2>/dev/null
echo "[VDN] Готово."
echo "Логи: $LOGS_DIR"
ls "$LOGS_DIR"/vdn_agent_*.csv 2>/dev/null
