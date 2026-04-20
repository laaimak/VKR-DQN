#include "RewardEvaluator.h"
#include <cmath>
#include <algorithm>

RewardEvaluator::RewardEvaluator(double gamma,
                                 double w1,
                                 double w2,
                                 double r_goal,
                                 double kickable_bonus,
                                 double own_half_penalty,
                                 int    agent_id)
    : M_gamma(gamma)
    , M_w1(w1)
    , M_w2(w2)
    , M_r_goal(r_goal)
    , M_kickable_bonus(kickable_bonus)
    , M_own_half_penalty(own_half_penalty)
    , M_agent_id(agent_id)
    , M_accumulated_reward(0.0)
    , M_current_tau(0)
    , M_last_distance(-1.0)
    , M_last_stamina(8000.0)
{}

void RewardEvaluator::startMacroAction(const rcsc::WorldModel& wm,
                                        const rcsc::Vector2D& target_pos)
{
    // Инициализируем базовые значения в начале макро-действия
    M_last_distance = wm.self().pos().dist(target_pos);
    M_last_stamina  = wm.self().stamina();
    M_accumulated_reward = 0.0;
    M_current_tau = 0;
}

void RewardEvaluator::updateStep(const rcsc::WorldModel& wm, const rcsc::Vector2D& target_pos)
{
    double current_distance = wm.self().pos().dist(target_pos);
    double current_stamina  = wm.self().stamina();
    double shaping_reward = 0.0;

    // Роль определяется по agent_id (AGENT_FORCE_ID), а не по server-unum.
    // В Stage 2 сервер раздаёт unum 2,3,4 — не соответствует реальной роли агента.
    const bool is_forward   = (M_agent_id >= 10);
    const bool is_defender  = (M_agent_id >= 2 && M_agent_id <= 5);
    // Полузащитник: agent_id 6-9

    double delta_dist    = M_last_distance - current_distance;
    double delta_stamina = std::max(0.0, M_last_stamina - current_stamina);

    // РОЛЬ: НАПАДАЮЩИЙ
    if (is_forward) {
        shaping_reward = M_w1 * delta_dist - M_w2 * delta_stamina;
        if (wm.self().isKickable()) shaping_reward += M_kickable_bonus;
        if (wm.self().isFrozen())   shaping_reward -= 10.0;
        if (wm.gameMode().type() == rcsc::GameMode::PlayOn && wm.ball().pos().x < 0.0) {
            shaping_reward -= M_own_half_penalty;
        }
    }
    // РОЛЬ: ЗАЩИТНИК
    else if (is_defender) {
        shaping_reward = M_w1 * delta_dist - M_w2 * delta_stamina;
        if (wm.self().isKickable()) {
            shaping_reward += M_kickable_bonus * 2.0;
        }
        double dist_to_our_goal = wm.ball().pos().dist(rcsc::Vector2D(-52.5, 0.0));
        if (dist_to_our_goal < 20.0) {
            shaping_reward -= 0.05;
        }
        if (wm.ball().pos().x > 0.0) {
            shaping_reward += 0.01;
        }
    }
    // РОЛЬ: ПОЛУЗАЩИТНИК
    else {
        shaping_reward = M_w1 * delta_dist - M_w2 * delta_stamina;
        if (wm.self().isKickable()) {
            shaping_reward += M_kickable_bonus;
        }
    }

    // Штраф за скученность: если союзник ближе 3м.
    // Для полузащитника уменьшен — они по природе ближе к центру поля.
    if (!is_forward) {
        const double crowd_penalty = is_defender ? 0.3 : 0.05;
        for (const rcsc::PlayerObject* tm : wm.teammates()) {
            if (tm && tm->unum() != 1 && tm->distFromSelf() < 3.0) {
                shaping_reward -= crowd_penalty;
            }
        }
    }

    shaping_reward = std::clamp(shaping_reward, -10.0, 10.0);

    M_accumulated_reward += std::pow(M_gamma, M_current_tau) * shaping_reward;
    M_last_distance = current_distance;
    M_last_stamina  = current_stamina;
    M_current_tau++;
}

double RewardEvaluator::terminalGoalReward(const rcsc::WorldModel& wm) const
{
    if (wm.gameMode().type() != rcsc::GameMode::AfterGoal_) {
        return 0.0;
    }

    if (wm.gameMode().side() == rcsc::NEUTRAL) {
        return 0.0;
    }

    const bool our_goal = (wm.gameMode().side() == wm.ourSide());

    // Используем agent_id для определения роли (не server-unum).
    if (M_agent_id >= 10) {
        // Нападающий: полная награда за гол, штраф за пропущенный
        return our_goal ? M_r_goal : -M_r_goal;
    }
    if (M_agent_id >= 2 && M_agent_id <= 5) {
        // Защитник: только штраф за пропущенный (мотивация держать оборону)
        return our_goal ? 0.0 : -M_r_goal;
    }
    if (M_agent_id >= 6 && M_agent_id <= 9) {
        // Полузащитник: частичная награда за гол, частичный штраф
        return our_goal ? 0.8 * M_r_goal : -0.8 * M_r_goal;
    }

    return our_goal ? M_r_goal : -M_r_goal;
}

void RewardEvaluator::addTerminalGoalReward(const rcsc::WorldModel& wm)
{
    M_accumulated_reward += std::pow(M_gamma, M_current_tau) * terminalGoalReward(wm);
}

double RewardEvaluator::getFinalRewardAndReset(int& out_tau)
{
    double final_reward  = M_accumulated_reward;
    out_tau              = M_current_tau;

    // Сброс для следующего макро-действия
    M_accumulated_reward = 0.0;
    M_current_tau        = 0;
    M_last_distance      = -1.0;

    return final_reward;
}

int RewardEvaluator::getCurrentTau() const
{
    return M_current_tau;
}