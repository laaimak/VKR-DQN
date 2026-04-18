#!/bin/bash

# ============================================================
# Скрипт запуска матча RoboCup
# Использование: ./start_match.sh [режим] [кол-во игроков]
#
# Режимы:
#   test    — helios vs helios (тестовый матч)
#   train   — DQN vs helios (тренировочный матч)
#   formation — показать расстановку всех 11 агентов на поле
#
# Примеры:
#   ./start_match.sh test 2     — 2 vs 2 (вратарь + 1 полевой)
#   ./start_match.sh train 2    — DQN 2 vs helios 2
#   ./start_match.sh train 11   — DQN 11 vs helios 11
# ============================================================

# --- Пути ---
HELIOS_ORIG="/Users/laaimak/Desktop/VKR/helios-original/build/bin/sample_player"
HELIOS_DQN="/Users/laaimak/Desktop/VKR/helios-base/src/player/sample_player"
HELIOS_DQN_DIR="/Users/laaimak/Desktop/VKR/helios-base/src"
HELIOS_ORIG_DIR="/Users/laaimak/Desktop/VKR/helios-original/build/bin"

HOST="localhost"
PORT="6000"

# --- Аргументы ---
MODE=${1:-"test"}     # test или train
N=${2:-2}             # количество игроков (включая вратаря)

echo "============================================"
echo " Режим: $MODE"
echo " Игроков в каждой команде: $N"
echo "============================================"

# --- Функция запуска команды ---
start_team() {
    local BINARY=$1
    local DIR=$2
    local TEAM=$3
    # local COUNT=$4

    #echo "[*] Запускаем команду $TEAM ($COUNT игроков)..."
    #for i in $(seq 1 $COUNT); do
        #cd "$DIR"
        #"$BINARY" --host $HOST --port $PORT -t "$TEAM" &
        #sleep 0.3
    #done
    #echo "[+] Команда $TEAM запущена."

    echo "[*] Запускаем команду $TEAM (Вратарь №1 и Агент №2)..."
    cd "$DIR"
    "$BINARY" --host $HOST --port $PORT -t "$TEAM" -n 1 -g &
    sleep 0.3
    "$BINARY" --host $HOST --port $PORT -t "$TEAM" -n 2 &
    sleep 0.3
}

# --- Запуск ---
if [ "$MODE" == "test" ]; then
    echo "[*] Тестовый матч: Helios_Opp vs Helios_Two"
    start_team "$HELIOS_ORIG" "$HELIOS_ORIG_DIR" "Helios_Opp" $N
    sleep 1
    start_team "$HELIOS_ORIG" "$HELIOS_ORIG_DIR" "Helios_Two" $N

elif [ "$MODE" == "train" ]; then
    echo "[*] Тренировочный матч: DQN_Team vs Helios_Opp"
    start_team "$HELIOS_DQN" "$HELIOS_DQN_DIR" "DQN_Team" $N
    sleep 1
    start_team "$HELIOS_ORIG" "$HELIOS_ORIG_DIR" "Helios_Opp" $N

elif [ "$MODE" == "formation" ]; then
    echo "[*] Режим расстановки: Выводим 11 оригинальных агентов на поле..."
    for i in $(seq 1 11); do
        cd "$HELIOS_ORIG_DIR"
        if [ $i -eq 1 ]; then
            "$HELIOS_ORIG" --host $HOST --port $PORT -t "Formation" -n 1 -g &
        else
            "$HELIOS_ORIG" --host $HOST --port $PORT -t "Formation" -n $i &
        fi
        sleep 0.3
    done

else
    echo "Неизвестный режим: $MODE"
    echo "Используй: ./start_match.sh [test|train] [кол-во игроков]"
    exit 1
fi

echo ""
echo "[*] Все агенты запущены. Ожидание матча..."
wait