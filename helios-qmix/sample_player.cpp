// -*-c++-*-

/*
 * Copyright: Hidehisa AKIYAMA
 * Modified for VDN integration: CTDE via Flask server (train) / pybind11 (inference)
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "sample_player.h"
#include "strategy.h"
#include "field_analyzer.h"
#include "action_chain_holder.h"
#include "sample_field_evaluator.h"
#include "soccer_role.h"
#include "sample_communication.h"
#include "keepaway_communication.h"
#include "sample_freeform_message_parser.h"
#include "bhv_penalty_kick.h"
#include "bhv_set_play.h"
#include "bhv_set_play_kick_in.h"
#include "bhv_set_play_indirect_free_kick.h"
#include "bhv_custom_before_kick_off.h"
#include "bhv_strict_check_shoot.h"
#include "view_tactical.h"
#include "intention_receive.h"

#include "basic_actions/basic_actions.h"
#include "basic_actions/bhv_emergency.h"
#include "basic_actions/body_go_to_point.h"
#include "basic_actions/body_intercept.h"
#include "basic_actions/body_kick_one_step.h"
#include "basic_actions/body_pass.h"
#include "basic_actions/body_advance_ball2009.h"
#include "basic_actions/body_clear_ball2009.h"
#include "basic_actions/body_hold_ball2008.h"
#include "basic_actions/neck_scan_field.h"
#include "basic_actions/neck_turn_to_ball_or_scan.h"
#include "basic_actions/view_synch.h"
#include "basic_actions/kick_table.h"

#include <rcsc/formation/formation.h>
#include <rcsc/player/intercept_table.h>
#include <rcsc/player/say_message_builder.h>
#include <rcsc/player/audio_sensor.h>
#include <rcsc/common/abstract_client.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/common/player_param.h>
#include <rcsc/common/audio_memory.h>
#include <rcsc/common/say_message_parser.h>
#include <rcsc/param/param_map.h>
#include <rcsc/param/cmd_line_parser.h>

// VDN модули
#include "dqn/vdn_bridge.h"
#include "dqn/StateBuilder.h"
#include "dqn/RewardEvaluator.h"

#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <algorithm>

using namespace rcsc;

// Пути к VDN конфигурации и Python модулям
static const std::string VDN_CONFIG_PATH = "/Users/laaimak/Desktop/VKR/python_vdn/config_vdn.json";
static const std::string VDN_MODULE_DIR  = "/Users/laaimak/Desktop/VKR/python_vdn";
static const std::string VDN_LOGS_PATH   = "/Users/laaimak/Desktop/VKR/helios-qmix/src/logs";

// Параметры вознаграждения
static const double REWARD_GAMMA          = 0.99;
static const double REWARD_W1             = 0.04;
static const double REWARD_W2             = 0.001;
static const double REWARD_GOAL           = 500.0;
static const double REWARD_KICKABLE_BONUS = 2.0;
static const double REWARD_OWN_HALF_PEN   = 0.0;

static const int GOALIE_UNUM = 1;


SamplePlayer::SamplePlayer()
    : PlayerAgent()
    , M_communication()
    , M_vdn_bridge(nullptr)
    , M_reward_evaluator(nullptr)
    , M_current_macro_action(0)
    , M_macro_action_timer(0)
    , M_last_state()
    , M_last_action(0)
    , M_macro_active(false)
    , M_first_action(true)
    , M_goal_event_consumed(false)
    , M_episode_reward(0.0)
{
    M_field_evaluator  = createFieldEvaluator();
    M_action_generator = createActionGenerator();

    std::shared_ptr<AudioMemory> audio_memory(new AudioMemory);
    M_worldmodel.setAudioMemory(audio_memory);

    addSayMessageParser(new BallMessageParser(audio_memory));
    addSayMessageParser(new PassMessageParser(audio_memory));
    addSayMessageParser(new InterceptMessageParser(audio_memory));
    addSayMessageParser(new GoalieMessageParser(audio_memory));
    addSayMessageParser(new GoalieAndPlayerMessageParser(audio_memory));
    addSayMessageParser(new OffsideLineMessageParser(audio_memory));
    addSayMessageParser(new DefenseLineMessageParser(audio_memory));
    addSayMessageParser(new WaitRequestMessageParser(audio_memory));
    addSayMessageParser(new PassRequestMessageParser(audio_memory));
    addSayMessageParser(new DribbleMessageParser(audio_memory));
    addSayMessageParser(new BallGoalieMessageParser(audio_memory));
    addSayMessageParser(new OnePlayerMessageParser(audio_memory));
    addSayMessageParser(new TwoPlayerMessageParser(audio_memory));
    addSayMessageParser(new ThreePlayerMessageParser(audio_memory));
    addSayMessageParser(new SelfMessageParser(audio_memory));
    addSayMessageParser(new TeammateMessageParser(audio_memory));
    addSayMessageParser(new OpponentMessageParser(audio_memory));
    addSayMessageParser(new BallPlayerMessageParser(audio_memory));
    addSayMessageParser(new StaminaMessageParser(audio_memory));
    addSayMessageParser(new RecoveryMessageParser(audio_memory));

    addFreeformMessageParser(
        new OpponentPlayerTypeMessageParser(M_worldmodel));

    M_communication = Communication::Ptr(new SampleCommunication());
}

SamplePlayer::~SamplePlayer()
{
    finalizeEpisode(false);

    if (VDNBridge::isPythonStarted()) {
        std::fflush(nullptr);
        std::_Exit(0);
    }
}


bool SamplePlayer::initImpl(CmdLineParser& cmd_parser)
{
    bool result = PlayerAgent::initImpl(cmd_parser);
    result &= Strategy::instance().init(cmd_parser);

    rcsc::ParamMap my_params("Additional options");
    cmd_parser.parse(my_params);

    if (cmd_parser.count("help") > 0) {
        my_params.printHelp(std::cout);
        return false;
    }

    if (cmd_parser.failed()) {
        std::cerr << "player: ***WARNING*** unsupported options: ";
        cmd_parser.print(std::cerr);
        std::cerr << std::endl;
    }

    if (!result) return false;

    if (!Strategy::instance().read(config().configDir())) {
        std::cerr << "***ERROR*** Failed to read team strategy." << std::endl;
        return false;
    }

    if (KickTable::instance().read(config().configDir() + "/kick-table")) {
        std::cerr << "Loaded kick table." << std::endl;
    }

    return true;
}

// Ленивая инициализация VDN Bridge
void SamplePlayer::initVDNIfNeeded()
{
    if (M_vdn_bridge) return;
    if (M_vdn_init_failed) return;

    // Вратарь управляется FSM helios
    if (world().self().unum() == GOALIE_UNUM) return;
    if (world().self().goalie()) return;

    int unum     = world().self().unum();
    int agent_id = unum;

    // Переопределение через переменную окружения
    if (const char* forced_id = std::getenv("AGENT_FORCE_ID")) {
        const int parsed = std::atoi(forced_id);
        if (parsed > 0) agent_id = parsed;
    }

    try {
        M_vdn_bridge = std::make_unique<VDNBridge>(
            agent_id, VDN_CONFIG_PATH, VDN_MODULE_DIR
        );

        M_reward_evaluator = std::make_unique<RewardEvaluator>(
            REWARD_GAMMA,
            REWARD_W1,
            REWARD_W2,
            REWARD_GOAL,
            REWARD_KICKABLE_BONUS,
            REWARD_OWN_HALF_PEN,
            agent_id
        );

        M_episode_reward  = 0.0;
        M_first_action    = true;
        M_episode_finalized = false;

        std::cerr << "[SamplePlayer] VDN initialized: unum=" << unum
                  << " agent_id=" << agent_id
                  << " mode=" << (M_vdn_bridge->isInference() ? "INFERENCE" : "TRAIN")
                  << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[SamplePlayer] VDN init failed: " << e.what() << std::endl;
        M_vdn_bridge.reset();
        M_vdn_init_failed = true;
    }
}

// Условия досрочного завершения макро-действия
bool SamplePlayer::isMacroActionDone(const WorldModel& wm) const
{
    switch (M_current_macro_action) {
    case 1: // shoot
        return !wm.self().isKickable();
    case 2: // pass
        return !wm.self().isKickable();
    case 3: // dribble
        if (!wm.self().isKickable() && wm.ball().distFromSelf() > 3.0)
            return true;
        return false;
    case 4: // clear
        return !wm.self().isKickable();
    case 5: // hold
        return false;
    case 6: // intercept
        return wm.self().isKickable();
    case 7: // block
        return wm.self().pos().dist(wm.ball().pos()) < 2.0;
    case 8: // positioning
        {
            Vector2D tactical_pos = Strategy::i().getPosition(wm.self().unum());
            return wm.self().pos().dist(tactical_pos) < 1.0;
        }
    default:
        return false;
    }
}

int SamplePlayer::getMaxTau(int action) const
{
    if (action >= 1 && action <= 8)
        return M_max_tau_by_action[static_cast<std::size_t>(action - 1)];
    return 20;
}

// Выполнение макро-действия
bool SamplePlayer::executeMacroAction(int action)
{
    const WorldModel& wm = world();
    bool executed = false;

    switch (action) {
    case 1: // shoot
        if (wm.self().isKickable()) {
            executed = Bhv_StrictCheckShoot().execute(this);
            if (!executed && wm.self().pos().dist(ServerParam::i().theirTeamGoalPos()) < 25.0) {
                executed = Body_KickOneStep(
                    ServerParam::i().theirTeamGoalPos(),
                    ServerParam::i().ballSpeedMax()
                ).execute(this);
            }
        } else {
            executed = Body_Intercept().execute(this);
        }
        break;

    case 2: // pass
        if (wm.self().isKickable()) {
            rcsc::Vector2D pass_point;
            double pass_speed = 0.0;
            int receiver = 0;
            if (Body_Pass::get_best_pass(wm, &pass_point, &pass_speed, &receiver)) {
                // Сообщаем получателю через say/hear
                if (effector().getSayMessageLength() == 0) {
                    if (effector().getSayMessageLength() == 0) {
                    addSayMessage(new PassMessage(
                        receiver,
                        pass_point,
                        effector().queuedNextBallPos(),
                        effector().queuedNextBallVel()
                    ));
                }
                }
                executed = Body_Pass().execute(this);
            } else {
                // Нет хорошего паса — держим мяч
                executed = Body_HoldBall2008().execute(this);
            }
        } else {
            executed = Body_Intercept().execute(this);
        }
        break;

    case 3: // dribble — ведём мяч в сторону ворот
        if (wm.self().isKickable()) {
            rcsc::Vector2D goal_dir =
                (ServerParam::i().theirTeamGoalPos() - wm.self().pos()).normalizedVector();
            rcsc::Vector2D dribble_target = wm.self().pos() + goal_dir * 5.0;
            executed = Body_KickOneStep(dribble_target, 1.2).execute(this);
        } else {
            executed = Body_Intercept().execute(this);
        }
        break;

    case 4: // clear
        if (wm.self().isKickable()) {
            executed = Body_ClearBall2009().execute(this);
        } else {
            executed = Body_Intercept().execute(this);
        }
        break;

    case 5: // hold
        if (wm.self().isKickable()) {
            executed = Body_HoldBall2008().execute(this);
        } else {
            executed = Body_Intercept().execute(this);
        }
        break;

    case 6: // intercept
        executed = Body_Intercept().execute(this);
        this->setNeckAction(new Neck_TurnToBall());
        break;

    case 7: // block
        executed = Body_GoToPoint(
            wm.ball().pos(), 0.5,
            ServerParam::i().maxDashPower()
        ).execute(this);
        break;

    case 8: // positioning
        executed = Body_GoToPoint(
            Strategy::i().getPosition(wm.self().unum()),
            0.5,
            ServerParam::i().maxDashPower()
        ).execute(this);
        break;

    default:
        {
            SoccerRole::Ptr role = Strategy::i().createRole(wm.self().unum(), wm);
            if (role) executed = role->execute(this);
        }
        break;
    }

    if (!executed) {
        executed = Body_Intercept().execute(this);
    }

    this->setNeckAction(new Neck_TurnToBallOrScan(0));
    return executed;
}

// Финализация эпизода
void SamplePlayer::finalizeEpisode(bool /*terminate_process*/)
{
    if (!M_vdn_bridge || M_episode_finalized) return;

    const WorldModel& wm = world();

    std::cerr << "--- VDN FINAL SAVING START ---" << std::endl;

    int our_score = (wm.ourSide() == rcsc::LEFT)
        ? wm.gameMode().scoreLeft()
        : wm.gameMode().scoreRight();
    int opp_score = (wm.ourSide() == rcsc::LEFT)
        ? wm.gameMode().scoreRight()
        : wm.gameMode().scoreLeft();

    const double goal_terminal = our_score * REWARD_GOAL - opp_score * REWARD_GOAL;

    // Последний переход: отправляем на сервер с done=true
    if (M_macro_active && M_reward_evaluator
        && !M_last_state.empty() && M_last_action > 0) {
        int    tau          = 0;
        double shaping      = M_reward_evaluator->getFinalRewardAndReset(tau);
        double total_reward = shaping + goal_terminal;

        const std::vector<double> final_state = StateBuilder::getState(wm);

        // TRAIN: отправляем на Flask; INFERENCE: игнорирует reward, делает forward pass
        M_vdn_bridge->step(final_state, total_reward, /*done=*/true);

        M_episode_reward += total_reward;

        std::cerr << "[VDN FINAL] shaping=" << shaping
                  << " goal=" << goal_terminal
                  << " total=" << total_reward
                  << " tau=" << tau << std::endl;
    } else {
        M_episode_reward += goal_terminal;
        std::cerr << "[VDN FINAL] goal_terminal=" << goal_terminal
                  << " our=" << our_score << " opp=" << opp_score << std::endl;
    }

    // Логируем эпизод в CSV
    const std::string log_path = VDN_LOGS_PATH
        + "/vdn_agent_" + std::to_string(M_vdn_bridge->agentId())
        + "_episode_rewards.csv";

    bool has_content = false;
    {
        std::ifstream f(log_path.c_str());
        has_content = f.good() && f.peek() != std::ifstream::traits_type::eof();
    }

    int match_id = 1;
    if (has_content) {
        std::ifstream f(log_path.c_str());
        std::string line;
        int last_id = 0;
        while (std::getline(f, line)) {
            if (line.empty() || !std::isdigit((unsigned char)line[0])) continue;
            std::size_t cp = line.find(',');
            if (cp == std::string::npos) continue;
            int id = std::atoi(line.substr(0, cp).c_str());
            if (id > last_id) last_id = id;
        }
        match_id = last_id + 1;
    }

    {
        std::ofstream f(log_path.c_str(), std::ios::app);
        if (!has_content) {
            f << "Match,EpisodeReward,ScoreFor,ScoreAgainst,StepsDone,Epsilon\n";
        }
        f << match_id << ','
          << std::fixed << std::setprecision(6) << M_episode_reward << ','
          << our_score << ','
          << opp_score << ','
          << M_vdn_bridge->stepsDone() << ','
          << std::fixed << std::setprecision(4) << M_vdn_bridge->epsilon() << '\n';
        f.flush();
    }

    std::cout << "Score: " << our_score << " - " << opp_score << std::endl;
    std::cout.flush();

    M_episode_finalized = true;

    std::fflush(nullptr);
    std::_Exit(0);
}

// Главный цикл принятия решений
void SamplePlayer::actionImpl()
{
    if (this->audioSensor().trainerMessageTime() == world().time()) {
        std::cerr << world().ourTeamName() << ' '
                  << world().self().unum() << ' '
                  << world().time()
                  << " trainer message["
                  << this->audioSensor().trainerMessage() << ']'
                  << std::endl;
    }

    Strategy::instance().update(world());
    FieldAnalyzer::instance().update(world());

    M_field_evaluator  = createFieldEvaluator();
    M_action_generator = createActionGenerator();

    ActionChainHolder::instance().setFieldEvaluator(M_field_evaluator);
    ActionChainHolder::instance().setActionGenerator(M_action_generator);

    if (doPreprocess()) {
        dlog.addText(Logger::TEAM, __FILE__": preprocess done");
        return;
    }

    ActionChainHolder::instance().update(world());

    SoccerRole::Ptr role_ptr;
    {
        role_ptr = Strategy::i().createRole(world().self().unum(), world());
        if (!role_ptr) {
            std::cerr << config().teamName() << ": "
                      << world().self().unum()
                      << " Error. Role not registered.\n";
            M_client->setServerAlive(false);
            return;
        }
    }

    // Set-play: стандартный helios
    if (world().gameMode().type() != GameMode::PlayOn
        && world().gameMode().type() != GameMode::AfterGoal_
        && role_ptr->acceptExecution(world())) {
        role_ptr->execute(this);
        return;
    }

    // Вратарь
    if (world().self().unum() == GOALIE_UNUM) {
        role_ptr->execute(this);
        return;
    }

    // VDN управление полевыми игроками в режиме PlayOn
    if (world().gameMode().type() == GameMode::PlayOn) {

        initVDNIfNeeded();

        if (!M_vdn_bridge) {
            role_ptr->execute(this);
            return;
        }

        const WorldModel& wm = world();

        Vector2D tactical_pos = Strategy::i().getPosition(wm.self().unum());
        Vector2D target_pos   = StateBuilder::getTargetPosition(
            wm, tactical_pos, M_vdn_bridge->agentId());

        bool done = (wm.time().cycle() >= M_match_end_cycle - 2);

        // Определяем нужно ли выбирать новое макро-действие
        bool macro_expired = M_macro_active
            && (isMacroActionDone(wm)
                || M_macro_action_timer >= getMaxTau(M_current_macro_action));

        bool select_new_action = !M_macro_active || M_first_action || macro_expired;

        if (select_new_action) {
            // Если предыдущее макро завершилось — отправляем переход на сервер
            if (macro_expired && !M_first_action) {
                int    tau          = 0;
                double final_reward = M_reward_evaluator->getFinalRewardAndReset(tau);
                M_episode_reward += final_reward;

                std::vector<double> next_state = StateBuilder::getState(wm);

                // TRAIN: отправляет (state, reward, done) на Flask → получает действие
                // INFERENCE: reward игнорируется, делает локальный argmax
                int new_action = M_vdn_bridge->step(next_state, final_reward, done);

                std::cerr << "[VDN] step. tau=" << tau
                          << " reward=" << final_reward
                          << " -> action=" << new_action << std::endl;

                M_current_macro_action = new_action;
                M_last_state           = next_state;
                M_last_action          = new_action;
            } else {
                // Первое действие эпизода: POST /reset → первое действие
                M_last_state = StateBuilder::getState(wm);
                int first_act = M_vdn_bridge->reset(M_last_state);

                M_current_macro_action = first_act;
                M_last_action          = first_act;
                M_first_action         = false;
                M_goal_event_consumed  = false;
            }

            M_macro_action_timer = 0;
            M_macro_active       = true;

            // Коммуникация: форвард слушает сигнал паса
            {
                const int aid = M_vdn_bridge->agentId();
                if (aid >= 10) {
                    if (M_pass_receive_timer > 0) {
                        M_current_macro_action = 6;
                        M_pass_receive_timer--;
                        std::cerr << "[COMM] Forward intercept timer="
                                  << M_pass_receive_timer << std::endl;
                    }
                    if (wm.audioMemory().passTime() == wm.time()
                        && !wm.audioMemory().pass().empty()) {
                        for (const auto& p : wm.audioMemory().pass()) {
                            if (p.receiver_ == wm.self().unum()) {
                                M_pass_receive_timer   = 5;
                                M_current_macro_action = 6;
                                std::cerr << "[COMM] Forward received PASS signal" << std::endl;
                                break;
                            }
                        }
                    }
                }
            }

            //Ролевые ограничения 
            {
                const int    aid      = M_vdn_bridge->agentId();
                const double ball_x   = wm.ball().pos().x;
                const bool   has_ball = wm.self().isKickable();

                // Защитник (2-5): мяч в чужой половине → только positioning/intercept/block
                if (aid >= 2 && aid <= 5 && ball_x > 0.0) {
                    if (M_current_macro_action == 0
                     || M_current_macro_action == 1
                     || M_current_macro_action == 2
                     || M_current_macro_action == 3
                     || M_current_macro_action == 5) {
                        M_current_macro_action = 8;
                    }
                }

                // Полузащитник (6-9): мяч глубоко у нас → только positioning/intercept
                if (aid >= 6 && aid <= 9 && ball_x < -25.0 && !has_ball) {
                    if (M_current_macro_action == 0
                     || M_current_macro_action == 1
                     || M_current_macro_action == 3
                     || M_current_macro_action == 5) {
                        M_current_macro_action = 8;
                    }
                }
            }

            // Без мяча: ударные действия → intercept
            if (!wm.self().isKickable()
                && (M_current_macro_action == 1
                    || M_current_macro_action == 2
                    || M_current_macro_action == 4)) {
                M_current_macro_action = 6;
            }

            M_last_action = M_current_macro_action;

            // Начинаем накопление награды для нового макро-действия
            M_reward_evaluator->startMacroAction(wm, target_pos);
        }

        // Обновляем шейпинговую награду за этот такт
        M_reward_evaluator->updateStep(wm, target_pos);
        M_macro_action_timer++;

        // Выполняем макро-действие
        executeMacroAction(M_current_macro_action);

        if (done) {
            finalizeEpisode(true);
            return;
        }

        return;
    }

    // Прочие режимы
    if (world().gameMode().isPenaltyKickMode()) {
        dlog.addText(Logger::TEAM, __FILE__": penalty kick");
        Bhv_PenaltyKick().execute(this);
        return;
    }

    Bhv_SetPlay().execute(this);
}

void SamplePlayer::handleActionStart()  {}
void SamplePlayer::handlePlayerType()   {}

void SamplePlayer::handleActionEnd()
{
    // Полевые VDN-агенты завершают эпизод в конце матча
    if (!M_episode_finalized
        && M_vdn_bridge
        && world().self().unum() != GOALIE_UNUM
        && world().time().cycle() >= M_match_end_cycle - 2) {
        finalizeEpisode(true);
        return;
    }

    // Вратарь тоже выходит одновременно с полевыми — не играет в одиночку
    if (world().self().goalie()
        && world().time().cycle() >= M_match_end_cycle - 2) {
        std::fflush(nullptr);
        std::_Exit(0);
    }

    if (world().self().posValid()) {
        debugClient().addLine(
            Vector2D(world().ourDefenseLineX(),
                     world().self().pos().y - 2.0),
            Vector2D(world().ourDefenseLineX(),
                     world().self().pos().y + 2.0));
        debugClient().addLine(
            Vector2D(world().offsideLineX(),
                     world().self().pos().y - 15.0),
            Vector2D(world().offsideLineX(),
                     world().self().pos().y + 15.0));
    }
}

void SamplePlayer::handleInitMessage()
{
    std::vector<int> pk_order = {10, 9, 2, 11, 3, 4, 1, 5, 6, 7, 8};
    M_worldmodel.setPenaltyKickTakerOrder(pk_order);

    initVDNIfNeeded();
}

void SamplePlayer::handleServerParam()
{
    if (ServerParam::i().keepawayMode()) {
        std::cerr << "set Keepaway mode communication." << std::endl;
        M_communication = Communication::Ptr(new KeepawayCommunication());
    }
}

void SamplePlayer::handlePlayerParam()
{
    if (KickTable::instance().createTables()) {
        std::cerr << world().teamName() << ' '
                  << world().self().unum()
                  << ": KickTable created." << std::endl;
    } else {
        std::cerr << world().teamName() << ' '
                  << world().self().unum()
                  << ": KickTable failed..." << std::endl;
        M_client->setServerAlive(false);
    }
}

void SamplePlayer::communicationImpl()
{
    if (M_communication) M_communication->execute(this);
}

bool SamplePlayer::doPreprocess()
{
    const WorldModel& wm = this->world();
    dlog.addText(Logger::TEAM, __FILE__": (doPreProcess)");

    if (wm.self().isFrozen()) {
        dlog.addText(Logger::TEAM, __FILE__": tackle wait. expires=%d",
                     wm.self().tackleExpires());
        this->setViewAction(new View_Tactical());
        this->setNeckAction(new Neck_TurnToBallOrScan(0));
        return true;
    }

    if (wm.gameMode().type() == GameMode::BeforeKickOff
        || wm.gameMode().type() == GameMode::AfterGoal_) {
        dlog.addText(Logger::TEAM, __FILE__": before_kick_off");
        Vector2D move_point = Strategy::i().getPosition(wm.self().unum());
        Bhv_CustomBeforeKickOff(move_point).execute(this);
        this->setViewAction(new View_Tactical());
        return true;
    }

    if (!wm.self().posValid()) {
        dlog.addText(Logger::TEAM, __FILE__": invalid my pos");
        Bhv_Emergency().execute(this);
        return true;
    }

    const int count_thr = (wm.self().goalie() ? 10 : 5);
    if (wm.ball().posCount() > count_thr
        || (wm.gameMode().type() != GameMode::PlayOn
            && wm.ball().seenPosCount() > count_thr + 10)) {
        dlog.addText(Logger::TEAM, __FILE__": search ball");
        this->setViewAction(new View_Tactical());
        Bhv_NeckBodyToBall().execute(this);
        return true;
    }

    this->setViewAction(new View_Tactical());

    if (doShoot())            return true;
    if (this->doIntention())  return true;
    if (doForceKick())        return true;
    if (doHeardPassReceive()) return true;

    return false;
}

bool SamplePlayer::doShoot()
{
    const WorldModel& wm = this->world();
    if (wm.gameMode().type() != GameMode::IndFreeKick_
        && wm.time().stopped() == 0
        && wm.self().isKickable()
        && Bhv_StrictCheckShoot().execute(this)) {
        dlog.addText(Logger::TEAM, __FILE__": shooted");
        this->setIntention(static_cast<SoccerIntention*>(0));
        return true;
    }
    return false;
}

bool SamplePlayer::doForceKick()
{
    const WorldModel& wm = this->world();
    if (wm.gameMode().type() == GameMode::PlayOn
        && !wm.self().goalie()
        && wm.self().isKickable()
        && wm.kickableOpponent()) {
        dlog.addText(Logger::TEAM, __FILE__": simultaneous kick");
        this->debugClient().addMessage("SimultaneousKick");
        Vector2D goal_pos(ServerParam::i().pitchHalfLength(), 0.0);
        if (wm.self().pos().x > 36.0 && wm.self().pos().absY() > 10.0)
            goal_pos.x = 45.0;
        Body_KickOneStep(goal_pos, ServerParam::i().ballSpeedMax()).execute(this);
        this->setNeckAction(new Neck_ScanField());
        return true;
    }
    return false;
}

bool SamplePlayer::doHeardPassReceive()
{
    const WorldModel& wm = this->world();
    if (wm.audioMemory().passTime() != wm.time()
        || wm.audioMemory().pass().empty()
        || wm.audioMemory().pass().front().receiver_ != wm.self().unum()) {
        return false;
    }

    int self_min       = wm.interceptTable().selfStep();
    Vector2D heard_pos = wm.audioMemory().pass().front().receive_pos_;

    if (!wm.kickableTeammate()
        && wm.ball().posCount() <= 1
        && wm.ball().velCount() <= 1
        && self_min < 20) {
        Body_Intercept().execute(this);
        this->setNeckAction(new Neck_TurnToBall());
    } else {
        Body_GoToPoint(heard_pos, 0.5,
                       ServerParam::i().maxDashPower()).execute(this);
        this->setNeckAction(new Neck_TurnToBall());
    }

    this->setIntention(new IntentionReceive(
        heard_pos, ServerParam::i().maxDashPower(),
        0.9, 5, wm.time()));
    return true;
}

FieldEvaluator::ConstPtr SamplePlayer::getFieldEvaluator() const
{
    return M_field_evaluator;
}

FieldEvaluator::ConstPtr SamplePlayer::createFieldEvaluator() const
{
    return FieldEvaluator::ConstPtr(new SampleFieldEvaluator);
}

#include "actgen_cross.h"
#include "actgen_direct_pass.h"
#include "actgen_self_pass.h"
#include "actgen_strict_check_pass.h"
#include "actgen_short_dribble.h"
#include "actgen_simple_dribble.h"
#include "actgen_shoot.h"
#include "actgen_action_chain_length_filter.h"

ActionGenerator::ConstPtr SamplePlayer::createActionGenerator() const
{
    CompositeActionGenerator* g = new CompositeActionGenerator();
    g->addGenerator(new ActGen_RangeActionChainLengthFilter(
        new ActGen_Shoot(), 2,
        ActGen_RangeActionChainLengthFilter::MAX));
    g->addGenerator(new ActGen_MaxActionChainLengthFilter(
        new ActGen_StrictCheckPass(), 1));
    g->addGenerator(new ActGen_MaxActionChainLengthFilter(
        new ActGen_Cross(), 1));
    g->addGenerator(new ActGen_MaxActionChainLengthFilter(
        new ActGen_ShortDribble(), 1));
    g->addGenerator(new ActGen_MaxActionChainLengthFilter(
        new ActGen_SelfPass(), 1));
    return ActionGenerator::ConstPtr(g);
}
