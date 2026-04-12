#include "RewardEvaluator.h"
#include <cmath>
#include <algorithm>

RewardEvaluator::RewardEvaluator(double gamma,
                                 double w1,
                                 double w2,
                                 double r_goal,
                                 double kickable_bonus)
    : M_gamma(gamma)
    , M_w1(w1)
    , M_w2(w2)
    , M_r_goal(r_goal)
    , M_kickable_bonus(kickable_bonus)
    , M_accumulated_reward(0.0)
    , M_current_tau(0)
    , M_last_distance(-1.0)
    , M_last_stamina(8000.0)
{}

void RewardEvaluator::startMacroAction(const rcsc::WorldModel& wm,
                                        const rcsc::Vector2D& target_pos)
{
    M_last_distance      = wm.self().pos().dist(target_pos);
    M_last_stamina       = wm.self().stamina();
    M_accumulated_reward = 0.0;
    M_current_tau        = 0;
}

void RewardEvaluator::updateStep(const rcsc::WorldModel& wm,
                                  const rcsc::Vector2D& target_pos)
{
    double current_distance = wm.self().pos().dist(target_pos);
    double current_stamina  = wm.self().stamina();

    // Потенциальная компонента: приближение к цели — награда, удаление — штраф
    double delta_dist    = M_last_distance - current_distance;
    double delta_stamina = std::max(0.0, M_last_stamina - current_stamina);
    double step_reward   = M_w1 * delta_dist - M_w2 * delta_stamina;

    // Бонус за владение мячом
    if (wm.self().isKickable()) {
        step_reward += M_kickable_bonus;
    }

    // Штраф за жёлтую карточку
    if (wm.self().isFrozen()) {
        step_reward -= 10.0;
    }

    // Штраф за красную карточку
    if (wm.self().tackleExpires() > 5) {
        step_reward -= 50.0;
    }

    // Терминальная компонента: голы
    rcsc::GameMode::Type game_type = wm.gameMode().type();
    if (game_type == rcsc::Goal_L || game_type == rcsc::Goal_R) {
        if (wm.gameMode().side() == wm.ourSide()) {
            step_reward += M_r_goal;
        } else if (wm.gameMode().side() != rcsc::NEUTRAL) {
            step_reward -= M_r_goal;
        }
    }

    // Дисконтированное накопление: R_t += gamma^tau * r_t
    M_accumulated_reward += std::pow(M_gamma, M_current_tau) * step_reward;

    M_last_distance = current_distance;
    M_last_stamina  = current_stamina;
    M_current_tau++;
}

double RewardEvaluator::getFinalRewardAndReset(int& out_tau)
{
    double final_reward  = M_accumulated_reward;
    out_tau              = M_current_tau;

    M_accumulated_reward = 0.0;
    M_current_tau        = 0;
    M_last_distance      = -1.0;

    return final_reward;
}

int RewardEvaluator::getCurrentTau() const
{
    return M_current_tau;
}