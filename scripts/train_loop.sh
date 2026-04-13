#!/bin/bash

# ============================================================
# Скрипт автоматического обучения агента DQN
# Использование: ./train_loop.sh [кол-во матчей]
# Пример: ./train_loop.sh 100
# ============================================================

source /Users/laaimak/Desktop/VKR/.venv314/bin/activate

MATCHES=${1:-100}

HELIOS_DQN="/Users/laaimak/Desktop/VKR/helios-base/src/player/sample_player"
HELIOS_DQN_DIR="/Users/laaimak/Desktop/VKR/helios-base/src"
HELIOS_OPP="/Users/laaimak/Desktop/VKR/helios-original/build/bin/sample_player"
HELIOS_OPP_DIR="/Users/laaimak/Desktop/VKR/helios-original/build/bin"
RCSSSERVER_DIR="/Users/laaimak/Desktop/VKR/rcssserver-master/build"

MATCH_DURATION=620

echo "============================================"
echo " Начинаем обучение: $MATCHES матчей"
echo "============================================"

for match in $(seq 1 $MATCHES); do
    echo ""
    echo "=== Матч $match из $MATCHES ==="

    # Принудительно убиваем все процессы
    echo "[*] Останавливаем старые процессы..."
    pkill -9 -f "sample_player" 2>/dev/null
    pkill -9 -f "rcssserver" 2>/dev/null
    pkill -9 -f "rcssmonitor" 2>/dev/null

    # Ждём пока все агенты точно отключились
    while ps aux | grep -q "[s]ample_player"; do
        pkill -9 -f "sample_player" 2>/dev/null
        echo "Ждём отключения агентов..."
        sleep 1
    done

    # Запускаем сервер
    echo "[*] Запускаем сервер..."
    cd "$RCSSSERVER_DIR"
    ./rcssserver server::auto_mode=true server::synch_mode=true &
    sleep 2

    # Запускаем DQN команду
    echo "[*] Запускаем DQN команду..."
    cd "$HELIOS_DQN_DIR"
    "$HELIOS_DQN" --host localhost --port 6000 -t DQN_Team -n 1 -g &
    sleep 0.2
    "$HELIOS_DQN" --host localhost --port 6000 -t DQN_Team -n 2 &
    sleep 0.2

    # Запускаем противника
    echo "[*] Запускаем противника..."
    cd "$HELIOS_OPP_DIR"
    "$HELIOS_OPP" --host localhost --port 6000 -t Helios_Opp -n 1 -g &
    sleep 0.2
    "$HELIOS_OPP" --host localhost --port 6000 -t Helios_Opp -n 11 &
    sleep 0.2

    # Целевой файл логов
    LOG_FILE="/Users/laaimak/Desktop/VKR/helios-base/src/logs/agent_2_steps.csv"
    # if [ ! -f "$LOG_FILE" ]; then
    #     echo "Step,Loss,AvgReward,Epsilon" > "$LOG_FILE"
    # fi

    echo "Матч идёт. Ждём завершения агентов..."
    WAITED=0
    while ps aux | grep -q "[s]ample_player" && [ $WAITED -lt $MATCH_DURATION ]; do
        sleep 1
        WAITED=$((WAITED + 1))

        if (( WAITED % 30 == 0 )); then
            echo "Прошло $WAITED сек... (Эпсилон всё еще трудится)"
        fi
    done

    # Принудительно убиваем после матча
    pkill -9 -f "sample_player" 2>/dev/null
    pkill -9 -f "rcssserver" 2>/dev/null
    sleep 1

    # Показываем прогресс
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