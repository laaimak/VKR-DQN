#!/bin/bash

# ============================================================
# Скрипт командного обучения DQN — Этап 2
# Запускает 4v4: агенты #2, #7, #11 + вратарь vs 4 helios
# Использование: ./train_team.sh [кол-во матчей]
# Пример: ./train_team.sh 300
# ============================================================

source /Users/laaimak/Desktop/VKR/.venv314/bin/activate

MATCHES=${1:-300}

HELIOS_DQN="/Users/laaimak/Desktop/VKR/helios-base/src/player/sample_player"
HELIOS_DQN_DIR="/Users/laaimak/Desktop/VKR/helios-base/src"
HELIOS_OPP="/Users/laaimak/Desktop/VKR/helios-original/build/bin/sample_player"
HELIOS_OPP_DIR="/Users/laaimak/Desktop/VKR/helios-original/build/bin"
RCSSSERVER_DIR="/Users/laaimak/Desktop/VKR/rcssserver-master/build"

MATCH_DURATION=800
LOGS_DIR="/Users/laaimak/Desktop/VKR/helios-base/src/logs"
TRAIN_LOGS_DIR="/Users/laaimak/Desktop/VKR/train_logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$TRAIN_LOGS_DIR"

SCORE_FILE="${TRAIN_LOGS_DIR}/team_score_${RUN_TS}.log"
echo "Командное обучение 4v4" > "$SCORE_FILE"
echo "Матч | Score | Агенты: #2 #7 #11" >> "$SCORE_FILE"

# Инициализируем CSV для каждого агента
for AGENT_ID in 2 7 11; do
    RESULTS_FILE="${LOGS_DIR}/agent_${AGENT_ID}_match_results.csv"
    if [ ! -f "$RESULTS_FILE" ] || [ ! -s "$RESULTS_FILE" ]; then
        echo "RunMatch,EpisodeId,ScoreFor,ScoreAgainst,EpisodeReward,StepsDone,Epsilon,Timestamp" > "$RESULTS_FILE"
    fi
done

echo "============================================"
echo " Командное обучение: $MATCHES матчей (4v4)"
echo " Агенты: #2 (защитник), #7 (хав), #11 (форвард)"
echo " Лог: $SCORE_FILE"
echo "============================================"

# Запоминаем последние EpisodeId для каждого агента
LAST_EP_2=""
LAST_EP_7=""
LAST_EP_11=""

for match in $(seq 1 $MATCHES); do
    echo ""
    echo "=== Матч $match из $MATCHES ==="

    # Останавливаем старые процессы
    pkill -9 -f "sample_player" 2>/dev/null
    pkill -9 -f "rcssserver" 2>/dev/null
    pkill -9 -f "rcssmonitor" 2>/dev/null

    while ps aux | grep -q "[s]ample_player"; do
        pkill -9 -f "sample_player" 2>/dev/null
        sleep 1
    done

    # Запускаем сервер
    cd "$RCSSSERVER_DIR"
    ./rcssserver server::auto_mode=true server::synch_mode=true \
        server::nr_extra_halfs=0 server::penalty_shoot_outs=false &
    sleep 2

    # Запускаем DQN команду (4 игрока)
    cd "$HELIOS_DQN_DIR"

    # Вратарь (#1) — helios FSM
    "$HELIOS_DQN" --host localhost --port 6000 -t DQN_Team -n 1 -g &
    sleep 0.3

    # Защитник (#2) — DQN
    AGENT_FORCE_ID=2 "$HELIOS_DQN" --host localhost --port 6000 -t DQN_Team -n 2 &
    sleep 0.3

    # Полузащитник (#7) — DQN
    AGENT_FORCE_ID=7 "$HELIOS_DQN" --host localhost --port 6000 -t DQN_Team -n 7 &
    sleep 0.3

    # Нападающий (#11) — DQN
    AGENT_FORCE_ID=11 "$HELIOS_DQN" --host localhost --port 6000 -t DQN_Team -n 11 &
    sleep 0.3

    # Запускаем противника (4 игрока helios)
    cd "$HELIOS_OPP_DIR"
    "$HELIOS_OPP" --host localhost --port 6000 -t Helios_Opp -n 1 -g &
    sleep 0.2
    "$HELIOS_OPP" --host localhost --port 6000 -t Helios_Opp -n 2 &
    sleep 0.2
    "$HELIOS_OPP" --host localhost --port 6000 -t Helios_Opp -n 7 &
    sleep 0.2
    "$HELIOS_OPP" --host localhost --port 6000 -t Helios_Opp -n 11 &
    sleep 0.2

    # Ждём завершения матча
    WAITED=0
    while ps aux | grep -q "[s]ample_player" && [ $WAITED -lt $MATCH_DURATION ]; do
        sleep 1
        WAITED=$((WAITED + 1))
        if (( WAITED % 30 == 0 )); then
            echo "Прошло $WAITED сек..."
        fi
    done

    pkill -TERM -f "sample_player" 2>/dev/null
    pkill -TERM -f "rcssserver" 2>/dev/null

    GRACE=0
    while ps aux | grep -q "[s]ample_player" && [ $GRACE -lt 30 ]; do
        sleep 1
        GRACE=$((GRACE + 1))
    done

    pkill -9 -f "sample_player" 2>/dev/null
    pkill -9 -f "rcssserver" 2>/dev/null
    sleep 1

    # Пишем результаты для каждого агента
    SCORE_FOR=""
    SCORE_AGAINST=""

    for AGENT_ID in 2 7 11; do
        EP_LOG="${LOGS_DIR}/agent_${AGENT_ID}_episode_rewards.csv"
        RESULTS_FILE="${LOGS_DIR}/agent_${AGENT_ID}_match_results.csv"

        eval "LAST_EP=\$LAST_EP_${AGENT_ID}"

        if [ -f "$EP_LOG" ] && [ "$(wc -l < "$EP_LOG")" -gt 1 ]; then
            LAST_LINE="$(tail -n 1 "$EP_LOG")"
            IFS=',' read -r EP_ID EP_REWARD SF SA STEPS EPS <<< "$LAST_LINE"

            if [ -n "$EP_ID" ] && [ "$EP_ID" != "$LAST_EP" ]; then
                if ! awk -F',' -v eid="$EP_ID" 'NR>1 && $2==eid {found=1} END{exit found ? 0 : 1}' "$RESULTS_FILE"; then
                    printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
                        "$match" "$EP_ID" "$SF" "$SA" "$EP_REWARD" "$STEPS" "$EPS" "$(date '+%Y-%m-%d %H:%M:%S')" \
                        >> "$RESULTS_FILE"
                    echo "[+] Агент #${AGENT_ID}: ${SF}-${SA} reward=${EP_REWARD} eps=${EPS}"
                fi
                eval "LAST_EP_${AGENT_ID}=$EP_ID"
                SCORE_FOR="$SF"
                SCORE_AGAINST="$SA"
            fi
        fi
    done

    echo "Матч $match | Score: ${SCORE_FOR}-${SCORE_AGAINST}" >> "$SCORE_FILE"
    echo "Матч $match завершён."
done

echo ""
echo "============================================"
echo " Командное обучение завершено! $MATCHES матчей."
echo "============================================"
