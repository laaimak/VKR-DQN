#include "RewardEvaluator.h"
#include <cmath>
#include <algorithm>

RewardEvaluator::RewardEvaluator(double gamma,
                                 double w1,
                                 double w2,
                                 double r_goal,
                                 double kickable_bonus,
                                 double own_half_penalty)
    : M_gamma(gamma)
    , M_w1(w1)
    , M_w2(w2)
    , M_r_goal(r_goal)
    , M_kickable_bonus(kickable_bonus)
    , M_own_half_penalty(own_half_penalty)
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
    int unum = wm.self().unum();

    // РОЛЬ: НАПАДАЮЩИЙ (Номера 10 и 11)
    if (unum == 10 || unum == 11) {
        double delta_dist = M_last_distance - current_distance;
        double delta_stamina = std::max(0.0, M_last_stamina - current_stamina);
        shaping_reward = M_w1 * delta_dist - M_w2 * delta_stamina;

        if (wm.self().isKickable()) shaping_reward += M_kickable_bonus;
        if (wm.self().isFrozen()) shaping_reward -= 10.0;
        
        // Штраф за свою половину поля (гоним его в атаку)
        if (wm.gameMode().type() == rcsc::GameMode::PlayOn && wm.ball().pos().x < 0.0) {
            shaping_reward -= M_own_half_penalty;
        }
    }
    // РОЛЬ: ЗАЩИТНИК (Номера 2, 3, 4, 5)
    else if (unum >= 2 && unum <= 5) {
        
        // 1. Награда за владение мячом (перехват/отбор)
        if (wm.self().isKickable()) {
            shaping_reward += M_kickable_bonus * 2.0; // Защитнику важнее отбирать мяч
        }
        
        // 2. Штраф, если мяч слишком близко к нашим воротам
        double dist_to_our_goal = wm.ball().pos().dist(rcsc::Vector2D(-52.5, 0.0));
        if (dist_to_our_goal < 20.0) {
            shaping_reward -= 0.05; // Бьем тревогой, мяч в опасной зоне!
        }
        
        // 3. Награда за вынос мяча
        // Для простоты пока даем бонус, если мяч перешел центр поля
        if (wm.ball().pos().x > 0.0) {
            shaping_reward += 0.01; // Мяч выбит, можно выдохнуть
        }
    }
    // РОЛЬ: ПОЛУЗАЩИТНИК (Номера 6, 7, 8, 9)
    else if (unum >= 6 && unum <= 9) {
        double delta_dist = M_last_distance - current_distance;
        double delta_stamina = std::max(0.0, M_last_stamina - current_stamina);

        // Полузащитники должны поддерживать переход фазы: прессинг + продвижение.
        shaping_reward = 0.7 * M_w1 * delta_dist - 0.8 * M_w2 * delta_stamina;

        if (wm.self().isKickable()) {
            shaping_reward += 0.5 * M_kickable_bonus;
        }
    }

    // Ограничиваем только shaping-компоненту, чтобы не резать терминальный сигнал r_goal.
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
    const int unum = wm.self().unum();

    if (unum == 10 || unum == 11) {
        return our_goal ? M_r_goal : -M_r_goal;
    }

    if (unum >= 2 && unum <= 5) {
        // Для защиты ключевой сигнал — сильный штраф за пропущенный мяч.
        return our_goal ? 0.0 : -M_r_goal;
    }

    if (unum >= 6 && unum <= 9) {
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