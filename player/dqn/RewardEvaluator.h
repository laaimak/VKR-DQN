#pragma once

#include <rcsc/player/world_model.h>
#include <rcsc/geom/vector_2d.h>

class RewardEvaluator {
public:
    RewardEvaluator(double gamma        = 0.99,
                    double w1           = 1.0,
                    double w2           = 0.001,
                    double r_goal       = 100.0,
                    double kickable_bonus = 5.0,
                    double own_half_penalty = 0.02);

    void startMacroAction(const rcsc::WorldModel& wm,
                           const rcsc::Vector2D& target_pos);

    void updateStep(const rcsc::WorldModel& wm,
                     const rcsc::Vector2D& target_pos);

    void addTerminalGoalReward(const rcsc::WorldModel& wm);

    double getFinalRewardAndReset(int& out_tau);

    int getCurrentTau() const;

    double terminalGoalReward(const rcsc::WorldModel& wm) const;

private:
    double M_gamma;
    double M_w1;
    double M_w2;
    double M_r_goal;
    double M_kickable_bonus;
    double M_own_half_penalty;

    double M_accumulated_reward;
    int    M_current_tau;
    double M_last_distance;
    double M_last_stamina;
};