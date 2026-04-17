#include "StateBuilder.h"
#include <algorithm>
#include <rcsc/player/world_model.h>
#include <cassert>

/**
 * Формирует вектор состояния s_t размерностью 18.
 *
 * Компоненты вектора:
 *
 *  [0]     d_b      — евклидово расстояние до мяча              [0, ~125]
 *  [1]     theta_b  — относительный угол до мяча                [-180, 180]
 *  [2]     d_g      — расстояние до ворот противника             [0, ~110]
 *  [3]     theta_g  — относительный угол до ворот               [-180, 180]
 *  [4]     x_a      — абсолютная координата X агента            [-52.5, 52.5]
 *  [5]     y_a      — абсолютная координата Y агента            [-34, 34]
 *  [6-15]  d_k, theta_k (K=5) — расстояния и углы до K ближайших игроков
 *  [16]    stam_t   — текущий запас выносливости                [0, 8000]
 *  [17]    v_t      — текущая скалярная скорость агента         [0, 1.2]
 *
 * Параметр K=5 фиксирован для обеспечения постоянной размерности
 * входного вектора нейронной сети: dim(s_t) = 18.
 */
std::vector<double> StateBuilder::getState(const rcsc::WorldModel& wm)
{
    std::vector<double> state;
    state.reserve(18);

    // --- [0-1] Мяч ---
    // Расстояние от агента до мяча
    state.push_back(wm.ball().distFromSelf());
    // Относительный угол до мяча от направления корпуса агента
    double ball_angle = (wm.ball().pos() - wm.self().pos()).th().degree()
                        - wm.self().body().degree();
    state.push_back(ball_angle);

    // --- [2-3] Ворота противника (X=52.5, Y=0.0) ---
    rcsc::Vector2D opp_goal(52.5, 0.0);
    state.push_back(wm.self().pos().dist(opp_goal));
    double goal_angle = (opp_goal - wm.self().pos()).th().degree()
                        - wm.self().body().degree();
    state.push_back(goal_angle);

    // --- [4-5] Абсолютные координаты агента ---
    state.push_back(wm.self().pos().x);
    state.push_back(wm.self().pos().y);

    // --- [6-15] K=5 ближайших игроков (союзники + противники) ---
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

    // Берём K=5 ближайших
    // Для игроков не обнаруженных в текущем такте — используем заглушки
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
            // Заглушка для ненаблюдаемого игрока
            state.push_back(125.0);
            state.push_back(0.0);
        }
    }

    // --- [16] Выносливость ---
    state.push_back(wm.self().stamina());

    // --- [17] Скалярная скорость ---
    state.push_back(wm.self().vel().r());

    // Проверка размерности
    // dim(s_t) должна быть строго равна 18
    assert(state.size() == 18);

    return state;
}

/**
 * Определяет тактическую цель агента для вычисления потенциальной
 * компоненты вознаграждения.
 *
 * Кусочно-заданная функция:
 *   target_i = C_ball, если i = argmin_j d(C_j, C_ball)  — ближайший к мячу
 *   target_i = P_i,    иначе                              — тактическая позиция
 *
 * где P_i — координаты тактической позиции i-го агента,
 * априорно заданные стратегией команды (из formations helios-base).
 */
rcsc::Vector2D StateBuilder::getTargetPosition(
    const rcsc::WorldModel& wm,
    const rcsc::Vector2D& tactical_pos)
{
    // Определяем расстояние текущего агента до мяча
    double my_dist = wm.self().pos().dist(wm.ball().pos());

    // Проверяем является ли текущий агент ближайшим к мячу среди союзников
    bool i_am_closest = true;
    for (const rcsc::PlayerObject* tm : wm.teammates()) {
        if (!tm) continue;
        if (tm->unum() == wm.self().unum()) continue;
        if (tm->unum() == 1) continue;

        if (tm->pos().dist(wm.ball().pos()) < my_dist) {
            i_am_closest = false;
            break;
        }
    }

    if (i_am_closest) {
        return wm.ball().pos();
    } else {
        return tactical_pos;
    }
}