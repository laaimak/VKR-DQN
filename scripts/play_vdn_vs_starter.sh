#!/bin/bash

# ============================================================
# Финальная игра VDN Team vs StarterAgent2D-V2
#
# Режим: INFERENCE (без обучения, без Flask)
# Коммуникация агентов: через say/hear сервера игры
# Скорость: обычный режим (synch_mode=false)
# Веса: vdn_results/vs_starter/weights/
#
# Использование:
#   ./play_vdn_vs_starter.sh
# ============================================================

LOG_FILE="/Users/laaimak/Desktop/VKR/play_vdn_vs_starter.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Запуск $(date) ===" > "$LOG_FILE"

VDN_BUILD="/Users/laaimak/Desktop/VKR/helios-qmix/build"
VDN_PLAYER="${VDN_BUILD}/sample_player"
VDN_PLAYER_CONF="${VDN_BUILD}/player.conf"
VDN_CONFIG_DIR="${VDN_BUILD}/formations-dt"

STARTER_DIR="/Users/laaimak/Desktop/VKR/StarterAgent2D-V2/build/bin"
STARTER_PLAYER="${STARTER_DIR}/sample_player"
STARTER_PLAYER_CONF="${STARTER_DIR}/player.conf"
STARTER_CONFIG_DIR="${STARTER_DIR}/formations-dt"

RCSSSERVER_DIR="/Users/laaimak/Desktop/VKR/rcssserver-master/build"

# Путь к обученным весам vs_starter
export VDN_WEIGHTS_DIR="/Users/laaimak/Desktop/VKR/vdn_results/vs_starter/weights"

# Inference mode: без Flask, без обучения, epsilon=0
export VDN_INFERENCE=1

echo "============================================================"
echo " VDN Team vs StarterAgent2D-V2 — ФИНАЛЬНАЯ ИГРА"
echo " Режим: INFERENCE (без обучения)"
echo " Веса:  ${VDN_WEIGHTS_DIR}"
echo " Коммуникация: say/hear сервера rcssserver"
echo "============================================================"

# Зачищаем остатки
pkill -9 -f "rcssserver" 2>/dev/null
pkill -9 -f "sample_player" 2>/dev/null
sleep 1

# Запускаем rcssserver в обычном (не ускоренном) режиме
cd "$RCSSSERVER_DIR"
./rcssserver \
    server::auto_mode=true \
    server::synch_mode=false \
    server::nr_normal_halfs=2 \
    server::nr_extra_halfs=0 \
    server::penalty_shoot_outs=false \
    server::half_time=300 \
    server::kick_off_wait=100 \
    &
RCSS_PID=$!
echo "[rcssserver] PID=$RCSS_PID"
sleep 2

# ─── VDN Team ────────────────────────────────────────────────
cd "$VDN_BUILD"
VDN_OPT="--player-config ${VDN_PLAYER_CONF} --config_dir ${VDN_CONFIG_DIR}"

echo "[VDN] Запускаем вратаря..."
"$VDN_PLAYER" $VDN_OPT --host localhost --port 6000 -t VDN_Team -g &
sleep 0.5

echo "[VDN] Запускаем полевых игроков 2-11..."
for unum in 2 3 4 5 6 7 8 9 10 11; do
    "$VDN_PLAYER" $VDN_OPT --host localhost --port 6000 -t VDN_Team &
    sleep 0.3
done

# ─── Противник: StarterAgent2D-V2 ───────────────────────────
echo "[Starter] Запускаем вратаря..."
OPP_OPT="--player-config ${STARTER_PLAYER_CONF} --config_dir ${STARTER_CONFIG_DIR}"
"$STARTER_PLAYER" $OPP_OPT --host localhost --port 6000 -t Starter -n 1 -g &
sleep 0.3

echo "[Starter] Запускаем полевых игроков 2-11..."
for unum in 2 3 4 5 6 7 8 9 10 11; do
    "$STARTER_PLAYER" $OPP_OPT --host localhost --port 6000 -t Starter &
    sleep 0.15
done

echo ""
echo "============================================================"
echo " Игра началась! Открой rcssmonitor для просмотра."
echo " Для остановки: Ctrl+C"
echo "============================================================"

# Ждём завершения игры
wait $RCSS_PID

echo ""
echo "=== Игра завершена ==="
pkill -TERM -f "sample_player" 2>/dev/null
sleep 2
pkill -9 -f "sample_player" 2>/dev/null
