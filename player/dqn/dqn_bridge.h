#pragma once

#include <vector>
#include <string>
#include <memory>

/**
 Мост между C++ агентом и Python/PyTorch нейросетью через pybind11.
 Каждый экземпляр соответствует одному агенту со своей сетью
 и буфером воспроизведения (Independent Q-Learning).
 */

class DQNBridge {
public:
    // agent_id:    номер агента 
    // config_path: путь к config.json
    // module_dir:  папка с agent.py, model.py, memory.py
    DQNBridge(int agent_id,
              const std::string& config_path,
              const std::string& module_dir);

    ~DQNBridge();

    // Выбор макро-действия по ε-жадной стратегии.
    // state: вектор состояния (18 признаков)
    // Возвращает action_id от 1 до 8
    int act(const std::vector<double>& state);

    // Сохранение перехода в буфер и один шаг обучения.
    // tau: длительность макро-действия в тактах
    void pushAndTrain(const std::vector<double>& state,
                      int action,
                      double reward,
                      const std::vector<double>& next_state,
                      bool done,
                      int tau);

    // Сохраняет веса лучшего агента по итогам матча.
    // all_rewards: {agent_id, episode_reward} для всех агентов
    void saveEliteIfBest(const std::vector<std::pair<int, double>>& all_rewards);

    // Загружает элитные веса как стартовую точку.
    void loadEliteWeights();

    // Сброс накопленной награды в начале нового матча.
    void resetEpisode();

    // Суммарная награда за текущий матч.
    double getEpisodeReward() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};