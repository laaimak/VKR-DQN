source /Users/laaimak/Desktop/VKR/.venv314/bin/activate

MATCHES=${1:-1000}
TRAIN_AGENT_ID=${2:-11}

HELIOS_DQN="/Users/laaimak/Desktop/VKR/helios-base/src/player/sample_player"
HELIOS_DQN_DIR="/Users/laaimak/Desktop/VKR/helios-base/src"
HELIOS_OPP="/Users/laaimak/Desktop/VKR/helios-original/build/bin/sample_player"
HELIOS_OPP_DIR="/Users/laaimak/Desktop/VKR/helios-original/build/bin"
RCSSSERVER_DIR="/Users/laaimak/Desktop/VKR/rcssserver-master/build"

MATCH_DURATION=800

TRAIN_LOGS_DIR="/Users/laaimak/Desktop/VKR/train_logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SCORE_ONLY_FILE="${TRAIN_LOGS_DIR}/agent_${TRAIN_AGENT_ID}_score_${RUN_TS}.log"

EPISODE_LOG_FILE="/Users/laaimak/Desktop/VKR/helios-base/src/logs/agent_${TRAIN_AGENT_ID}_episode_rewards.csv"
MATCH_RESULTS_FILE="/Users/laaimak/Desktop/VKR/helios-base/src/logs/agent_${TRAIN_AGENT_ID}_match_results.csv"
LAST_EPISODE_ID=""

mkdir -p "$TRAIN_LOGS_DIR"
{
    echo "Агент #$TRAIN_AGENT_ID"
    echo "Матч | Score"
} > "$SCORE_ONLY_FILE"

if [ ! -f "$MATCH_RESULTS_FILE" ] || [ ! -s "$MATCH_RESULTS_FILE" ]; then
    echo "RunMatch,EpisodeId,ScoreFor,ScoreAgainst,EpisodeReward,StepsDone,Epsilon,Timestamp" > "$MATCH_RESULTS_FILE"
fi

if [ -f "$EPISODE_LOG_FILE" ] && [ "$(wc -l < "$EPISODE_LOG_FILE")" -gt 1 ]; then
    LAST_EPISODE_ID="$(tail -n 1 "$EPISODE_LOG_FILE" | cut -d',' -f1)"
fi

echo "============================================"
echo " Начинаем обучение: $MATCHES матчей"
echo " Обучаемый агент: #$TRAIN_AGENT_ID"
echo " Лог счета матчей: $SCORE_ONLY_FILE"
echo "============================================"

for match in $(seq 1 $MATCHES); do
    echo ""
    echo "=== Матч $match из $MATCHES ==="

    echo "[*] Останавливаем старые процессы..."
    pkill -9 -f "sample_player" 2>/dev/null
    pkill -9 -f "rcssserver" 2>/dev/null
    pkill -9 -f "rcssmonitor" 2>/dev/null

    while ps aux | grep -q "[s]ample_player"; do
        pkill -9 -f "sample_player" 2>/dev/null
        echo "Ждём отключения агентов..."
        sleep 1
    done

    echo "[*] Запускаем сервер..."
    cd "$RCSSSERVER_DIR"
    ./rcssserver server::auto_mode=true server::synch_mode=true \
        server::nr_extra_halfs=0 server::penalty_shoot_outs=false &
    sleep 2

    echo "[*] Запускаем DQN команду..."
    cd "$HELIOS_DQN_DIR"
    "$HELIOS_DQN" --host localhost --port 6000 -t DQN_Team -n 1 -g &
    sleep 0.2

    export AGENT_FORCE_ID="$TRAIN_AGENT_ID"
    "$HELIOS_DQN" --host localhost --port 6000 -t DQN_Team -n 2 &
    sleep 0.2

    echo "[*] Запускаем противника..."
    cd "$HELIOS_OPP_DIR"
    "$HELIOS_OPP" --host localhost --port 6000 -t Helios_Opp -n 1 -g &
    sleep 0.2
    "$HELIOS_OPP" --host localhost --port 6000 -t Helios_Opp -n 3 &
    sleep 0.2

    LOG_FILE="/Users/laaimak/Desktop/VKR/helios-base/src/logs/agent_${TRAIN_AGENT_ID}_steps.csv"

    echo "Матч идёт. Ждём завершения агентов..."
    WAITED=0
    while ps aux | grep -q "[s]ample_player" && [ $WAITED -lt $MATCH_DURATION ]; do
        sleep 1
        WAITED=$((WAITED + 1))

        if (( WAITED % 30 == 0 )); then
            echo "Прошло $WAITED сек... (Эпсилон всё еще трудится)"
        fi
    done

    pkill -TERM -f "sample_player" 2>/dev/null
    pkill -TERM -f "rcssserver" 2>/dev/null

    GRACE_WAIT=0
    while ps aux | grep -q "[s]ample_player" && [ $GRACE_WAIT -lt 30 ]; do
        sleep 1
        GRACE_WAIT=$((GRACE_WAIT + 1))
    done

    pkill -9 -f "sample_player" 2>/dev/null
    pkill -9 -f "rcssserver" 2>/dev/null
    sleep 1

    if [ -f "$EPISODE_LOG_FILE" ] && [ "$(wc -l < "$EPISODE_LOG_FILE")" -gt 1 ]; then
        LAST_EPISODE_LINE="$(tail -n 1 "$EPISODE_LOG_FILE")"
        IFS=',' read -r EPISODE_ID EPISODE_REWARD SCORE_FOR SCORE_AGAINST STEPS_DONE EPSILON_VALUE <<< "$LAST_EPISODE_LINE"

        if [ -n "$EPISODE_ID" ] && [ "$EPISODE_ID" != "$LAST_EPISODE_ID" ]; then
            if ! awk -F',' -v eid="$EPISODE_ID" 'NR>1 && $2==eid {found=1} END{exit found ? 0 : 1}' "$MATCH_RESULTS_FILE"; then
                printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
                    "$match" "$EPISODE_ID" "$SCORE_FOR" "$SCORE_AGAINST" "$EPISODE_REWARD" "$STEPS_DONE" "$EPSILON_VALUE" "$(date '+%Y-%m-%d %H:%M:%S')" \
                    >> "$MATCH_RESULTS_FILE"
                echo "[+] Результат матча: ${SCORE_FOR}-${SCORE_AGAINST} (agent #$TRAIN_AGENT_ID)"
                echo "Матч $EPISODE_ID | Score: $SCORE_FOR - $SCORE_AGAINST" >> "$SCORE_ONLY_FILE"
            else
                echo "[!] Пропуск дубликата: EpisodeId=$EPISODE_ID уже есть в $MATCH_RESULTS_FILE"
                echo "Матч $EPISODE_ID | Score: $SCORE_FOR - $SCORE_AGAINST" >> "$SCORE_ONLY_FILE"
            fi
            LAST_EPISODE_ID="$EPISODE_ID"
        else
            echo "[!] Предупреждение: новый итог матча не появился в episode CSV"
        fi
    else
        echo "[!] Предупреждение: файл эпизодов пока пустой ($EPISODE_LOG_FILE)"
    fi

    if [ -f "$LOG_FILE" ]; then
        echo "[+] Последнее: $(tail -1 $LOG_FILE)"
        echo "[+] Всего шагов: $(wc -l < $LOG_FILE)"
    fi

    echo "Матч $match завершён."
done

echo ""
echo "============================================"
echo " Обучение завершено! $MATCHES матчей сыграно."
echo "============================================"