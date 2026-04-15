#pragma once

#include <vector>
#include <rcsc/player/world_model.h>
#include <rcsc/geom/vector_2d.h>

/**
 * StateBuilder — модуль формирования вектора состояния.
 *
 * Выполняет парсинг визуальных сообщений симулятора rcsoccersim
 * и формирует вещественный вектор состояния s_t фиксированной
 * размерности dim(s_t) = 18.
 */
class StateBuilder {
public:
    /**
     * Формирует вектор состояния s_t размерностью 18.
     * wm: модель мира helios-base (WorldModel)
     */
    static std::vector<double> getState(const rcsc::WorldModel& wm);

    /**
     * Определяет тактическую цель агента для Reward Shaping.
     * wm:           модель мира
     * tactical_pos: тактическая позиция P_i данного агента
     */
    static rcsc::Vector2D getTargetPosition(
        const rcsc::WorldModel& wm,
        const rcsc::Vector2D& tactical_pos);
};