#include "StateBuilder.h"
#include <algorithm>
#include <rcsc/player/world_model.h>
#include <cassert>

/**
 Формирует вектор состояния s_t размерностью 18.
 * Компоненты вектора:
[0] евклидово расстояние до мяча
[1] относительный угол до мяча
[2] расстояние до ворот противника
[3] относительный угол до ворот
[4] абсолютная координата X агента
[5] абсолютная координата Y агента
[6-15] расстояния и углы до K ближайших игроков
[16] текущий запас выносливости
[17] текущая скалярная скорость агента
 */

std::vector<double> StateBuilder::getState(const rcsc::WorldModel& wm)
{
    std::vector<double> state;
    state.reserve(18);

    // [0-1] Мяч
    // Расстояние от агента до мяча
    state.push_back(wm.ball().distFromSelf());
    // Относительный угол до мяча от направления корпуса агента
    double ball_angle = (wm.ball().pos() - wm.self().pos()).th().degree()
                        - wm.self().body().degree();
    state.push_back(ball_angle);

    // [2-3] Ворота противника
    rcsc::Vector2D opp_goal(52.5, 0.0);
    state.push_back(wm.self().pos().dist(opp_goal));
    double goal_angle = (opp_goal - wm.self().pos()).th().degree()
                        - wm.self().body().degree();
    state.push_back(goal_angle);

    // [4-5] Абсолютные координаты агента
    state.push_back(wm.self().pos().x);
    state.push_back(wm.self().pos().y);

    // [6-15] 3 ближайших партнёра + 2 ближайших противника

    // Партнёры по команде (исключаем себя и вратаря #1)
    std::vector<const rcsc::PlayerObject*> teammates;
    teammates.reserve(10);
    for (const rcsc::PlayerObject* p : wm.teammates()) {
        if (p && p->unum() != 1) {
            teammates.push_back(p);
        }
    }
    std::sort(teammates.begin(), teammates.end(),
        [](const rcsc::PlayerObject* a, const rcsc::PlayerObject* b) {
            return a->distFromSelf() < b->distFromSelf();
        });

    // Противники
    std::vector<const rcsc::PlayerObject*> opponents;
    opponents.reserve(11);
    for (const rcsc::PlayerObject* p : wm.opponents()) {
        if (p) opponents.push_back(p);
    }
    std::sort(opponents.begin(), opponents.end(),
        [](const rcsc::PlayerObject* a, const rcsc::PlayerObject* b) {
            return a->distFromSelf() < b->distFromSelf();
        });

    // 3 ближайших партнёра [6-11]
    for (int i = 0; i < 3; ++i) {
        if (i < static_cast<int>(teammates.size())
            && teammates[i]->distFromSelf() < 125.0) {
            state.push_back(teammates[i]->distFromSelf());
            double angle =
                (teammates[i]->pos() - wm.self().pos()).th().degree()
                - wm.self().body().degree();
            state.push_back(angle);
        } else {
            state.push_back(125.0);
            state.push_back(0.0);
        }
    }

    // 2 ближайших противника [12-15]
    for (int i = 0; i < 2; ++i) {
        if (i < static_cast<int>(opponents.size())
            && opponents[i]->distFromSelf() < 125.0) {
            state.push_back(opponents[i]->distFromSelf());
            double angle =
                (opponents[i]->pos() - wm.self().pos()).th().degree()
                - wm.self().body().degree();
            state.push_back(angle);
        } else {
            state.push_back(125.0);
            state.push_back(0.0);
        }
    }

    // [16] Выносливость
    state.push_back(wm.self().stamina());

    // [17] Скалярная скорость
    state.push_back(wm.self().vel().r());

    // Проверка размерности
    assert(state.size() == 18);

    return state;
}

  // Определяет тактическую цель агента для вычисления потенциальной
rcsc::Vector2D StateBuilder::getTargetPosition(
    const rcsc::WorldModel& wm,
    const rcsc::Vector2D& tactical_pos,
    int agent_id)
{
    const rcsc::Vector2D ball_pos = wm.ball().pos();
    const double my_dist = wm.self().pos().dist(ball_pos);

    const bool is_forward   = (agent_id >= 10);
    const bool is_midfielder = (agent_id >= 6 && agent_id <= 9);
    const bool is_defender  = (agent_id >= 2 && agent_id <= 5);

    // Нападающий: зональное позиционирование для атаки
    if (is_forward) {
        if (ball_pos.x < -20.0) {
            // Мяч в нашей половине — форвард держится ближе к центру, готов к приёму
            return rcsc::Vector2D(15.0, 0.0);
        } else if (ball_pos.x < 0.0) {
            // Мяч в центре — форвард немного впереди, готов ворваться
            return rcsc::Vector2D(25.0, 0.0);
        } else {
            // Мяч в чужой половине — прессинг и удар
            return ball_pos;
        }
    }

    // Защитник: строго держит линию обороны
    if (is_defender) {
        if (ball_pos.x > 0.0) {
            // Мяч в чужой половине — защитник не уходит вперёд
            return tactical_pos;
        }
    }

    // Полузащитник: не бросает оборону при глубокой атаке противника
    if (is_midfielder) {
        if (ball_pos.x < -25.0) {
            // Мяч глубоко в нашей половине — хав помогает обороне
            return tactical_pos;
        }
    }

    // Защитник и хав (когда мяч близко/в нашей половине):
    // идём к мячу если мы ближайший полевой игрок
    bool i_am_closest = true;
    for (const rcsc::PlayerObject* tm : wm.teammates()) {
        if (!tm) continue;
        if (tm->unum() == wm.self().unum()) continue;
        if (tm->unum() == 1) continue;
        if (tm->pos().dist(ball_pos) < my_dist) {
            i_am_closest = false;
            break;
        }
    }

    return i_am_closest ? ball_pos : tactical_pos;
}