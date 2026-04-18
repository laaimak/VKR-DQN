// -*-c++-*-

/*
 * Copyright: Hidehisa AKIYAMA
 * Modified for DQN integration
 */

#ifndef SAMPLE_PLAYER_H
#define SAMPLE_PLAYER_H

#include "action_generator.h"
#include "field_evaluator.h"
#include "communication.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/geom/vector_2d.h>

#include "dqn/RewardEvaluator.h"
#include "dqn/dqn_bridge.h"

#include <vector>
#include <memory>
#include <array>

// Forward declarations
class DQNBridge;

class SamplePlayer
    : public rcsc::PlayerAgent {
private:

    Communication::Ptr        M_communication;
    FieldEvaluator::ConstPtr  M_field_evaluator;
    ActionGenerator::ConstPtr M_action_generator;

    // DQN модуль — интегрирован как разделяемая библиотека через pybind11
    // Каждый экземпляр SamplePlayer имеет собственный DQNBridge — парадигма IQL
    std::unique_ptr<DQNBridge>      M_dqn_bridge;
    std::unique_ptr<RewardEvaluator> M_reward_evaluator;

    // Состояние макро-действия
    int                  M_current_macro_action; // текущее макро-действие (1-8)
    int                  M_macro_action_timer;   // счётчик тактов tau
    std::vector<double>  M_last_state;           // s_t на момент выбора действия
    int                  M_last_action;          // выбранное действие o_t
    bool                 M_macro_active;         // флаг активного макро-действия
    std::array<int, 8>   M_max_tau_by_action{{10, 10, 20, 10, 15, 30, 20, 40}};
    int                  M_match_end_cycle = 1200;
    bool                 M_episode_finalized = false;
    bool                 M_goal_event_consumed = false;
    bool M_dqn_init_failed = false;

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

    // Ленивая инициализация DQN (вызывается при первом такте PlayOn)
    void initDQNIfNeeded();

    // Условие досрочного завершения макро-действия
    bool isMacroActionDone(const rcsc::WorldModel& wm) const;

    // Максимальная длительность макро-действия (в тактах)
    int getMaxTau(int action) const;

    // Выполнение выбранного макро-действия
    // Возвращает true, если удалось выставить body-команду
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