#pragma once

#include <vector>
#include <string>
#include <memory>
#include <array>

/**
 DQNBridge — связующий программный интерфейс между базовым агентом C++ 
 и вычислительным графом нейронной сети Python/PyTorch.
 * Каждый экземпляр DQNBridge соответствует одному независимому агенту
 * и содержит собственный SMDPAgent с отдельной нейронной сетью и
 * буфером воспроизведения — реализует парадигму IQL.
 */
class DQNBridge {
public:

    DQNBridge(int agent_id,
              const std::string& config_path,
              const std::string& module_dir);

    ~DQNBridge();

    // Выбор макро-действия по ε-жадной стратегии.
    int act(const std::vector<double>& state);

    // Сохранение перехода и запуск одного шага обучения.
    void pushAndTrain(const std::vector<double>& state,
                      int action,
                      double reward,
                      const std::vector<double>& next_state,
                      bool done,
                      int tau);

    // Загрузка весов элитного агента в начале нового матча.
    void loadEliteWeights();

    // Сброс накопленного вознаграждения в начале нового матча.
    void resetEpisode();

    // Получение накопленного вознаграждения за текущий матч.
    double getEpisodeReward() const;

    // Логирование итога матча
    void logEpisodeResult(int our_score, int opp_score);

    // Сохранение лучших весов если текущий эпизод побил рекорд.
    void saveRecordIfBest();

    // Конфигурация runtime.
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