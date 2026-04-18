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

// DQN модули
#include "dqn/dqn_bridge.h"
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

static const std::string DQN_CONFIG_PATH = "/Users/laaimak/Desktop/VKR/python_dqn/config_defender.json";
static const std::string DQN_MODULE_DIR  = "/Users/laaimak/Desktop/VKR/python_dqn";

static const int GOALIE_UNUM = 1;


SamplePlayer::SamplePlayer()
    : PlayerAgent()
    , M_communication()
    , M_dqn_bridge(nullptr)
    , M_reward_evaluator(nullptr)
    , M_current_macro_action(0)
    , M_macro_action_timer(0)
    , M_last_state()
    , M_last_action(0)
    , M_macro_active(false)
    , M_goal_event_consumed(false)
{
    M_field_evaluator = createFieldEvaluator();
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

    if (KickTable::instance().read(
            config().configDir() + "/kick-table")) {
        std::cerr << "Loaded kick table." << std::endl;
    }

    return true;
}

// Инициализация DQN Bridge при первом такте PlayOn
void SamplePlayer::initDQNIfNeeded()
{
    if (M_dqn_bridge) return;
    if (M_dqn_init_failed) return; 

    if (world().ourTeamName() != "DQN_Team") return;

    if (world().self().unum() == GOALIE_UNUM) return;
    if (world().self().goalie()) return;

    int unum = world().self().unum();
    int agent_id = unum;

    if (const char* forced_id = std::getenv("AGENT_FORCE_ID")) {
        const int parsed = std::atoi(forced_id);
        if (parsed > 0) {
            agent_id = parsed;
        }
    }

    try {
        M_dqn_bridge = std::make_unique<DQNBridge>(
            agent_id, DQN_CONFIG_PATH, DQN_MODULE_DIR
        );

        M_max_tau_by_action = M_dqn_bridge->maxTauByAction();
        M_match_end_cycle = M_dqn_bridge->matchEndCycle();
        M_reward_evaluator = std::make_unique<RewardEvaluator>(
            M_dqn_bridge->rewardGamma(),
            M_dqn_bridge->rewardW1(),
            M_dqn_bridge->rewardW2(),
            M_dqn_bridge->rewardGoal(),
            M_dqn_bridge->rewardKickableBonus(),
            M_dqn_bridge->rewardOwnHalfPenalty()
        );
        M_dqn_bridge->resetEpisode();
        M_episode_finalized = false;
        
        std::cerr << "[SamplePlayer] DQN initialized: unum=" << unum
              << " agent_id=" << agent_id << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[SamplePlayer] DQN init failed: " << e.what() << std::endl;
        M_dqn_bridge.reset();
        M_dqn_init_failed = true;
    }
}

// Условия досрочного завершения макро-действия
bool SamplePlayer::isMacroActionDone(const WorldModel& wm) const
{
    switch (M_current_macro_action) {
    case 1: // shoot — завершается когда мяч покидает зону владения
        return !wm.self().isKickable();

    case 2: // pass — завершается когда мяч покидает зону владения
        return !wm.self().isKickable();

    case 3: // dribble — завершается когда агент переместился на >2м
            //           или потерял мяч
        if (!wm.self().isKickable()
            && wm.ball().distFromSelf() > 3.0) {
            return true;
        }
        return false;

    case 4: // clear — завершается когда мяч покидает зону владения
        return !wm.self().isKickable();

    case 5: // hold — завершается по таймеру (задаётся max_tau)
        return false;

    case 6: // intercept — завершается когда агент стал владеть мячом
        return wm.self().isKickable();

    case 7: // block — завершается когда агент встал между противником
            //         и воротами
        return wm.self().pos().dist(wm.ball().pos()) < 2.0;

    case 8: // positioning — завершается когда агент достиг позиции P_i
        {
            Vector2D tactical_pos =
                Strategy::i().getPosition(wm.self().unum());
            return wm.self().pos().dist(tactical_pos) < 1.0;
        }

    default:
        return false;
    }
}

// Максимальная длительность каждого макро-действия
int SamplePlayer::getMaxTau(int action) const
{
    if (action >= 1 && action <= 8) {
        return M_max_tau_by_action[static_cast<std::size_t>(action - 1)];
    }
    return 20;
}

// Выполнение макро-действия
bool SamplePlayer::executeMacroAction(int action)
{
    const WorldModel& wm = world();
    bool executed = false;

    switch (action) {
    case 1: // shoot
        if ( wm.self().isKickable() ) {
            executed = Bhv_StrictCheckShoot().execute(this);
        } else {
            executed = Body_Intercept().execute(this);
        }
        break;

    case 2: // pass
        if ( wm.self().isKickable() ) {
            executed = Body_KickOneStep(ServerParam::i().theirTeamGoalPos(),
                                        ServerParam::i().ballSpeedMax()).execute(this);
        } else {
            executed = Body_Intercept().execute(this);
        }
        break;

    case 3: // dribble
        if ( wm.self().isKickable() ) {
            executed = Body_AdvanceBall2009().execute(this);
        } else {
            executed = Body_Intercept().execute(this);
        }
        break;

    case 4: // clear
        if ( wm.self().isKickable() ) {
            executed = Body_ClearBall2009().execute(this);
        } else {
            executed = Body_Intercept().execute(this);
        }
        break;

    case 5: // hold
        if ( wm.self().isKickable() ) {
            executed = Body_HoldBall2008().execute(this);
        } else {
            executed = Body_Intercept().execute(this);
        }
        break;

    case 6: // intercept
        executed = Body_Intercept().execute(this);
        this->setNeckAction(new Neck_TurnToBall());
        break;

    case 7: // block — движемся к мячу чтобы блокировать противника
        executed = Body_GoToPoint(
            wm.ball().pos(), 0.5,
            ServerParam::i().maxDashPower()
        ).execute(this);
        break;

    case 8: // positioning — движемся на тактическую позицию
        executed = Body_GoToPoint(
            Strategy::i().getPosition(wm.self().unum()),
            0.5,
            ServerParam::i().maxDashPower()
        ).execute(this);
        break;

    default:
        // Fallback: стандартное поведение helios FSM
        {
            SoccerRole::Ptr role =
                Strategy::i().createRole(wm.self().unum(), wm);
            if (role) executed = role->execute(this);
        }
        break;
    }

    // Если выбранное поведение не выставило body-команду, даем безопасный fallback.
    if (!executed) {
        executed = Body_Intercept().execute(this);
    }

    this->setNeckAction( new Neck_TurnToBallOrScan( 0 ) );

    return executed;
}

// Финализация эпизода и запись итогов матча
void SamplePlayer::finalizeEpisode(bool terminate_process)
{
    if (!M_dqn_bridge || M_episode_finalized) {
        return;
    }

    const WorldModel& wm = world();

    std::cerr << "--- FINAL SAVING START ---" << std::endl;

    int our_score = (wm.ourSide() == rcsc::LEFT)
        ? wm.gameMode().scoreLeft()
        : wm.gameMode().scoreRight();
    int opp_score = (wm.ourSide() == rcsc::LEFT)
        ? wm.gameMode().scoreRight()
        : wm.gameMode().scoreLeft();

    const double r_goal = M_dqn_bridge->rewardGoal();
    const double goal_terminal = (our_score - opp_score) * r_goal;

    // Последний переход: шейпинг + терминальная награда за голы → в replay buffer
    if (M_macro_active && M_reward_evaluator
        && !M_last_state.empty() && M_last_action > 0) {
        int tau = 0;
        const double shaping_reward = M_reward_evaluator->getFinalRewardAndReset(tau);
        const double total_final_reward = shaping_reward + goal_terminal;
        const std::vector<double> current_state = StateBuilder::getState(wm);
        M_dqn_bridge->pushAndTrain(
            M_last_state, M_last_action, total_final_reward, current_state, true, tau);
        M_dqn_bridge->addEpisodeReward(total_final_reward);
        std::cerr << "[FINAL] pushAndTrain: shaping=" << shaping_reward
                  << " goal=" << goal_terminal
                  << " total=" << total_final_reward
                  << " tau=" << tau << std::endl;
    } else {
        // Макро не активно — только логируем goal_terminal
        M_dqn_bridge->addEpisodeReward(goal_terminal);
        std::cerr << "[FINAL] goal_terminal=" << goal_terminal
                  << " our=" << our_score << " opp=" << opp_score << std::endl;
    }

    const double episode_reward = M_dqn_bridge->getEpisodeReward();

    const std::string episode_log_path =
        M_dqn_bridge->logsPath()
        + "/" + "agent_"
        + std::to_string(M_dqn_bridge->agentId())
        + "_episode_rewards.csv";

    bool file_has_content = false;
    {
        std::ifstream check_file(episode_log_path.c_str());
        file_has_content = check_file.good() && check_file.peek() != std::ifstream::traits_type::eof();
    }

    int match_id = 1;
    if (file_has_content) {
        std::ifstream in_file(episode_log_path.c_str());
        std::string line;
        int last_match_id = 0;
        while (std::getline(in_file, line)) {
            if (line.empty()) {
                continue;
            }
            if (!std::isdigit(static_cast<unsigned char>(line[0]))) {
                continue;
            }
            const std::size_t comma_pos = line.find(',');
            if (comma_pos == std::string::npos) {
                continue;
            }
            const int parsed_id = std::atoi(line.substr(0, comma_pos).c_str());
            if (parsed_id > last_match_id) {
                last_match_id = parsed_id;
            }
        }
        match_id = last_match_id + 1;
    }

    {
        std::ofstream out_file(episode_log_path.c_str(), std::ios::app);
        if (!file_has_content) {
            out_file << "Match,EpisodeReward,ScoreFor,ScoreAgainst,StepsDone,Epsilon\n";
        }
        out_file << match_id << ','
                 << std::fixed << std::setprecision(6) << episode_reward << ','
                 << our_score << ','
                 << opp_score << ','
                 << M_dqn_bridge->stepsDone() << ','
                 << std::fixed << std::setprecision(4) << M_dqn_bridge->epsilon() << '\n';
        out_file.flush();
    }

    std::cout << "Score: " << our_score << " - " << opp_score << std::endl;
    std::cout.flush();

    M_dqn_bridge->saveRecordIfBest();

    std::cerr << "--- DISCONNECTING ---" << std::endl;

    M_episode_finalized = true;

    if (terminate_process) {
        std::fflush(nullptr);
        std::_Exit(0);
    }
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

    // Создаём роль для стандартного поведения
    SoccerRole::Ptr role_ptr;
    {
        role_ptr = Strategy::i().createRole(
            world().self().unum(), world());

        if (!role_ptr) {
            std::cerr << config().teamName() << ": "
                      << world().self().unum()
                      << " Error. Role not registered.\n";
            M_client->setServerAlive(false);
            return;
        }
    }

    // Режим set-play: используем стандартный helios
    // AfterGoal_ исключаем — он обрабатывается ниже в goal-блоке DQN
    if (world().gameMode().type() != GameMode::PlayOn
        && world().gameMode().type() != GameMode::AfterGoal_
        && role_ptr->acceptExecution(world())) {
        role_ptr->execute(this);
        return;
    }

    // Вратарь: всегда стандартный helios FSM
    if (world().self().unum() == GOALIE_UNUM) {
        role_ptr->execute(this);
        return;
    }

    // DQN управление полевыми игроками в режиме PlayOn
    if (world().gameMode().type() == GameMode::PlayOn) {

        // Инициализация DQN при первом такте
        initDQNIfNeeded();

        if (!M_dqn_bridge) {
            role_ptr->execute(this);
            return;
        }

        const WorldModel& wm = world();

        // Тактическая позиция текущего агента из formations helios-base
        Vector2D tactical_pos =
            Strategy::i().getPosition(wm.self().unum());

        // Цель для вычисления потенциала вознаграждения
        // (ближайший к мячу → мяч, остальные → тактическая позиция)
        Vector2D target_pos =
            StateBuilder::getTargetPosition(wm, tactical_pos);

        // Конец матча
        bool done = (wm.time().cycle() >= M_match_end_cycle - 2);

        // Проверяем завершение текущего макро-действия
        bool macro_done = !M_macro_active
            || isMacroActionDone(wm)
            || (M_macro_action_timer >= getMaxTau(M_current_macro_action));

        bool transition_pushed_this_cycle = false;

        if (macro_done && M_macro_active && !done) {
            // Получаем итоговую накопленную награду R_t и длительность tau
            int    tau          = 0;
            double final_reward = M_reward_evaluator->getFinalRewardAndReset(tau);

            // Текущий вектор состояния s_{t+tau}
            std::vector<double> current_state = StateBuilder::getState(wm);

            // Сохраняем переход и выполняем шаг обучения
            M_dqn_bridge->pushAndTrain(
                M_last_state,
                M_last_action,
                final_reward,
                current_state,
                false,
                tau
            );
            transition_pushed_this_cycle = true;
            std::cerr << "[DEBUG] pushAndTrain called. tau=" << tau
                    << " reward=" << final_reward << std::endl;
        }

        if (macro_done || !M_macro_active) {
            // Формируем новый вектор состояния s_t
            M_last_state = StateBuilder::getState(wm);

            // Выбираем новое макро-действие по ε-жадной стратегии
            M_current_macro_action = M_dqn_bridge->act(M_last_state);
            if (!wm.self().isKickable()
                && (M_current_macro_action == 1
                    || M_current_macro_action == 2
                    || M_current_macro_action == 4)) {
                // Без владения мячом ударные макро-действия вырождаются в tau=1,
                M_current_macro_action = 6;
            }
            M_last_action          = M_current_macro_action;
            M_macro_action_timer   = 0;
            M_macro_active         = true;
            M_goal_event_consumed  = false;

            // Инициализируем накопление награды для нового макро-действия
            M_reward_evaluator->startMacroAction(wm, target_pos);
        }

        // Обновляем награду за текущий такт
        M_reward_evaluator->updateStep(wm, target_pos);
        M_macro_action_timer++;

        // Выполняем выбранное макро-действие
        executeMacroAction(M_current_macro_action);

        // Обрабатываем конец матча
        if (done) {
            finalizeEpisode(true);
            return;
        }
        
        return;
    }


    // Прочие режимы (penalty kick, set play)
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
    if (!M_episode_finalized
        && M_dqn_bridge
        && world().self().unum() != GOALIE_UNUM
        && world().time().cycle() >= M_match_end_cycle - 2) {
        finalizeEpisode(true);
        return;
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

    if (doShoot())           return true;
    if (this->doIntention()) return true;
    if (doForceKick())       return true;
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
        if (wm.self().pos().x > 36.0 && wm.self().pos().absY() > 10.0) {
            goal_pos.x = 45.0;
        }
        Body_KickOneStep(goal_pos,
                         ServerParam::i().ballSpeedMax()).execute(this);
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

    int self_min = wm.interceptTable().selfStep();
    Vector2D intercept_pos = wm.ball().inertiaPoint(self_min);
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