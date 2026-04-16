#pragma once

#include <vector>
#include <string>
#include <memory>
#include <array>

/**
 * DQNBridge — связующий программный интерфейс между базовым агентом C++
 * и вычислительным графом нейронной сети Python/PyTorch.
 *
 * Реализует внутрипроцессный вызов через pybind11:
 * интеллектуальный модуль интегрируется непосредственно в адресное
 * пространство каждого C++ клиента в виде разделяемой библиотеки.
 * Это устраняет необходимость в межпроцессном взаимодействии (IPC)
 * и сериализации данных.
 *
 * Каждый экземпляр DQNBridge соответствует одному независимому агенту
 * и содержит собственный SMDPAgent с отдельной нейронной сетью и
 * буфером воспроизведения — реализует парадигму IQL.
 */
class DQNBridge {
public:
    /**
     * Конструктор.
     * agent_id:    номер агента (uniform_number из helios, 1-11)
     * config_path: путь к config.json
     * module_dir:  директория где лежат agent.py, model.py, memory.py
     */
    DQNBridge(int agent_id,
              const std::string& config_path,
              const std::string& module_dir);

    ~DQNBridge();

    /**
     * Выбор макро-действия по ε-жадной стратегии.
     * state: вектор состояния s^i_t размерностью 18
     * Возвращает action_id от 1 до 8
     */
    int act(const std::vector<double>& state);

    /**
     * Сохранение перехода и запуск одного шага обучения.
     * state:      вектор состояния s^i_t
     * action:     выбранное макро-действие (1-8)
     * reward:     накопленная дисконтированная награда R^i_t
     * next_state: вектор состояния s^i_{t+tau}
     * done:       флаг терминального состояния
     * tau:        длительность макро-действия
     */
    void pushAndTrain(const std::vector<double>& state,
                      int action,
                      double reward,
                      const std::vector<double>& next_state,
                      bool done,
                      int tau);

    /**
     * Parameter Sharing: сравнение вознаграждений всех агентов.
     * Вызывается по завершении матча.
     * all_rewards: вектор пар {agent_id, episode_reward}
     */
    void saveEliteIfBest(const std::vector<std::pair<int, double>>& all_rewards);

    /**
     * Загрузка весов элитного агента в начале нового матча.
     */
    void loadEliteWeights();

    /**
     * Сброс накопленного вознаграждения в начале нового матча.
     */
    void resetEpisode();

    /**
     * Получение накопленного вознаграждения за текущий матч.
     */
    double getEpisodeReward() const;

    /**
     * Логирование итога матча (счет + эпизодическая награда).
     */
    void logEpisodeResult(int our_score, int opp_score);

    /**
     * Сохранение лучших весов если текущий эпизод побил рекорд.
     * Вызывается в конце каждого матча из finalizeEpisode().
     */
    void saveRecordIfBest();

    // Конфигурация runtime, считанная из config.json через Python agent.
    double rewardGamma() const;
    double rewardW1() const;
    double rewardW2() const;
    double rewardGoal() const;
    double rewardKickableBonus() const;
    double rewardOwnHalfPenalty() const;
    std::array<int, 8> maxTauByAction() const;
    int matchEndCycle() const;
    const std::string& logsPath() const;
    void addEpisodeReward(double reward);
    int stepsDone() const;
    double epsilon() const;
    int agentId() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};