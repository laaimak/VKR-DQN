#!/bin/bash
echo "Останавливаем все процессы..."
pkill -f "sample_player"
pkill -f "rcssserver"
pkill -f "rcssmonitor"
sleep 1

pkill -9 -f "sample_player"
pkill -9 -f "rcssserver"
pkill -9 -f "rcssmonitor"
echo "Готово!"

#/Users/laaimak/Desktop/VKR/stop_all.sh