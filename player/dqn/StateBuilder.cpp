#include "StateBuilder.h"
#include <algorithm>
#include <rcsc/player/world_model.h>
#include <cassert>

// Вектор состояния s_t (18 признаков):
//  [0-1]   расстояние и угол до мяча
//  [2-3]   расстояние и угол до ворот противника
//  [4-5]   координаты X, Y агента
//  [6-15]  расстояния и углы до 5 ближайших игроков
//  [16]    выносливость
//  [17]    скорость агента
std::vector<double> StateBuilder::getState(const rcsc::WorldModel& wm)
{
    std::vector<double> state;
    state.reserve(18);

    // Мяч
    state.push_back(wm.ball().distFromSelf());
    double ball_angle = (wm.ball().pos() - wm.self().pos()).th().degree()
                        - wm.self().body().degree();
    state.push_back(ball_angle);

    // Ворота противника
    rcsc::Vector2D opp_goal(52.5, 0.0);
    state.push_back(wm.self().pos().dist(opp_goal));
    double goal_angle = (opp_goal - wm.self().pos()).th().degree()
                        - wm.self().body().degree();
    state.push_back(goal_angle);

    // Позиция агента
    state.push_back(wm.self().pos().x);
    state.push_back(wm.self().pos().y);

    // K=5 ближайших игроков
    std::vector<const rcsc::AbstractPlayerObject*> players;
    players.reserve(21);

    for (const rcsc::AbstractPlayerObject* p : wm.allPlayers()) {
        if (p && p->unum() != wm.self().unum()) {
            players.push_back(p);
        }
    }

    std::sort(players.begin(), players.end(),
        [](const rcsc::AbstractPlayerObject* a,
           const rcsc::AbstractPlayerObject* b) {
            return a->distFromSelf() < b->distFromSelf();
        });

    const int K = 5;
    for (int i = 0; i < K; ++i) {
        if (i < static_cast<int>(players.size())
            && players[i]->distFromSelf() < 125.0) {
            state.push_back(players[i]->distFromSelf());
            double player_angle =
                (players[i]->pos() - wm.self().pos()).th().degree()
                - wm.self().body().degree();
            state.push_back(player_angle);
        } else {
            // Игрок не виден — заглушка
            state.push_back(125.0);
            state.push_back(0.0);
        }
    }

    state.push_back(wm.self().stamina());
    state.push_back(wm.self().vel().r());

    assert(state.size() == 18);
    return state;
}

// Ближайший к мячу агент преследует мяч,
// остальные держат тактическую позицию P_i.
rcsc::Vector2D StateBuilder::getTargetPosition(
    const rcsc::WorldModel& wm,
    const rcsc::Vector2D& tactical_pos)
{
    double my_dist    = wm.self().pos().dist(wm.ball().pos());
    bool i_am_closest = true;

    for (const rcsc::PlayerObject* tm : wm.teammates()) {
        if (!tm) continue;
        if (tm->unum() == wm.self().unum()) continue;
        if (tm->unum() == 1) continue; // вратарь не в счёт

        if (tm->pos().dist(wm.ball().pos()) < my_dist) {
            i_am_closest = false;
            break;
        }
    }

    return i_am_closest ? wm.ball().pos() : tactical_pos;
}