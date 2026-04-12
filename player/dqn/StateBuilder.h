#pragma once

#include <vector>
#include <rcsc/player/world_model.h>
#include <rcsc/geom/vector_2d.h>

// Формирует вектор состояния s_t (18 признаков) из модели мира helios.
class StateBuilder {
public:
    // Возвращает вектор состояния размерностью 18.
    static std::vector<double> getState(const rcsc::WorldModel& wm);

    // Определяет цель агента для reward shaping:
    // ближайший к мячу → преследует мяч, остальные → тактическая позиция P_i.
    static rcsc::Vector2D getTargetPosition(
        const rcsc::WorldModel& wm,
        const rcsc::Vector2D& tactical_pos);
};