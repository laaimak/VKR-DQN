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
    double step_reward = 0.0;
    
    int unum = wm.self().unum();

    // РОЛЬ: НАПАДАЮЩИЙ (Номера 10 и 11)
    if (unum == 10 || unum == 11) {
        double delta_dist = M_last_distance - current_distance;
        double delta_stamina = std::max(0.0, M_last_stamina - current_stamina);
        step_reward = M_w1 * delta_dist - M_w2 * delta_stamina;

        if (wm.self().isKickable()) step_reward += M_kickable_bonus;
        if (wm.self().isFrozen()) step_reward -= 10.0;
        
        // Штраф за свою половину поля (гоним его в атаку)
        if (wm.gameMode().type() == rcsc::GameMode::PlayOn && wm.ball().pos().x < 0.0) {
            step_reward -= 0.02;
        }

        // Награда за гол
        if (wm.gameMode().type() == rcsc::Goal_L || wm.gameMode().type() == rcsc::Goal_R) {
            if (wm.gameMode().side() == wm.ourSide()) step_reward += M_r_goal;
            else if (wm.gameMode().side() != rcsc::NEUTRAL) step_reward -= M_r_goal;
        }
    }
    // РОЛЬ: ЗАЩИТНИК (Номера 2, 3, 4, 5)
    else if (unum >= 2 && unum <= 5) {
        
        // 1. Награда за владение мячом (перехват/отбор)
        if (wm.self().isKickable()) {
            step_reward += M_kickable_bonus * 2.0; // Защитнику важнее отбирать мяч
        }
        
        // 2. Штраф, если мяч слишком близко к нашим воротам
        double dist_to_our_goal = wm.ball().pos().dist(rcsc::Vector2D(-52.5, 0.0));
        if (dist_to_our_goal < 20.0) {
            step_reward -= 0.05; // Бьем тревогой, мяч в опасной зоне!
        }
        
        // 3. Награда за вынос мяча
        // Для простоты пока даем бонус, если мяч перешел центр поля
        if (wm.ball().pos().x > 0.0) {
            step_reward += 0.01; // Мяч выбит, можно выдохнуть
        }

        // 4. Огромный штраф за пропущенный гол
        if (wm.gameMode().type() == rcsc::Goal_L || wm.gameMode().type() == rcsc::Goal_R) {
            if (wm.gameMode().side() != wm.ourSide() && wm.gameMode().side() != rcsc::NEUTRAL) {
                step_reward -= M_r_goal; // Вся вина на защите!
            }
        }
    }

    M_accumulated_reward += std::pow(M_gamma, M_current_tau) * step_reward;
    M_last_distance = current_distance;
    M_last_stamina  = current_stamina;
    M_current_tau++;
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