#pragma once

#include <vector>
#include <rcsc/player/world_model.h>
#include <rcsc/geom/vector_2d.h>

// StateBuilder — модуль формирования вектора состояния.
class StateBuilder {
public:

    static std::vector<double> getState(const rcsc::WorldModel& wm);

    static rcsc::Vector2D getTargetPosition(
        const rcsc::WorldModel& wm,
        const rcsc::Vector2D& tactical_pos,
        int agent_id = 0);
};