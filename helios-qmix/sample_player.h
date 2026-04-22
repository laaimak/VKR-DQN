// -*-c++-*-

/*
 * Copyright: Hidehisa AKIYAMA
 * Modified for VDN integration (CTDE via Flask server)
 */

#ifndef SAMPLE_PLAYER_H
#define SAMPLE_PLAYER_H

#include "action_generator.h"
#include "field_evaluator.h"
#include "communication.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/geom/vector_2d.h>

#include "dqn/RewardEvaluator.h"
#include "dqn/vdn_bridge.h"

#include <vector>
#include <memory>
#include <array>

class SamplePlayer
    : public rcsc::PlayerAgent {
private:

    Communication::Ptr        M_communication;
    FieldEvaluator::ConstPtr  M_field_evaluator;
    ActionGenerator::ConstPtr M_action_generator;

    // VDN Bridge: TRAIN mode → HTTP к Flask; INFERENCE mode → локальная Q-сеть
    std::unique_ptr<VDNBridge>       M_vdn_bridge;
    std::unique_ptr<RewardEvaluator> M_reward_evaluator;

    // Состояние макро-действия
    int                  M_current_macro_action;
    int                  M_macro_action_timer;
    std::vector<double>  M_last_state;
    int                  M_last_action;
    bool                 M_macro_active;
    bool                 M_first_action;
    std::array<int, 8>   M_max_tau_by_action{{10, 10, 20, 10, 15, 30, 20, 40}};
    int                  M_match_end_cycle = 6000;
    bool                 M_episode_finalized = false;
    bool                 M_goal_event_consumed = false;
    bool                 M_vdn_init_failed = false;

    // Накопленная награда за эпизод (для логирования)
    double               M_episode_reward = 0.0;

    // Коммуникационный протокол pass/hear
    int  M_pass_receive_timer = 0;

public:

    SamplePlayer();
    virtual ~SamplePlayer();

protected:

    virtual bool initImpl(rcsc::CmdLineParser& cmd_parser);

    virtual void actionImpl();
    virtual void communicationImpl();

    virtual void handleActionStart();
    virtual void handleActionEnd();
    virtual void handleInitMessage();
    virtual void handleServerParam();
    virtual void handlePlayerParam();
    virtual void handlePlayerType();

    virtual FieldEvaluator::ConstPtr  createFieldEvaluator() const;
    virtual ActionGenerator::ConstPtr createActionGenerator() const;

private:

    void initVDNIfNeeded();

    bool isMacroActionDone(const rcsc::WorldModel& wm) const;
    int  getMaxTau(int action) const;
    bool executeMacroAction(int action);

    void finalizeEpisode(bool terminate_process);

    bool doPreprocess();
    bool doShoot();
    bool doForceKick();
    bool doHeardPassReceive();

public:

    virtual FieldEvaluator::ConstPtr getFieldEvaluator() const;
};

#endif
