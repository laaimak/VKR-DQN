#include "dqn_bridge.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

// Глобальный интерпретатор Python — создаётся один раз на весь процесс
// Используем счётчик ссылок чтобы не создавать повторно
static int s_interpreter_ref_count = 0;
static py::scoped_interpreter* s_interpreter = nullptr;

struct DQNBridge::Impl {
    int         agent_id;
    std::string config_path;
    std::string elite_weights_path;
    py::object  agent;       // экземпляр SMDPAgent
    double      episode_reward = 0.0;

    Impl(int agent_id,
         const std::string& config_path,
         const std::string& module_dir)
        : agent_id(agent_id)
        , config_path(config_path)
    {
        // Инициализируем интерпретатор Python один раз
        if (s_interpreter_ref_count == 0) {
            s_interpreter = new py::scoped_interpreter{};
        }
        s_interpreter_ref_count++;

        try {
            // Добавляем директорию с Python модулями в sys.path
            // Добавляем пути venv314
            py::module_ sys = py::module_::import("sys");
            sys.attr("path").attr("insert")(0, "/Users/laaimak/Desktop/VKR/.venv314/lib/python3.14/site-packages");
            sys.attr("path").attr("insert")(0, module_dir);

            // Импортируем модуль агента
            py::module_ agent_module = py::module_::import("agent");

            // Создаём экземпляр SMDPAgent для данного агента
            // Каждый агент имеет свой независимый экземпляр — парадигма IQL
            agent = agent_module.attr("SMDPAgent")(agent_id, config_path);

            // Читаем путь к файлу элитных весов из конфига
            py::module_ json_mod = py::module_::import("json");
            py::object  cfg_file = py::module_::import("builtins")
                                        .attr("open")(config_path, "r");
            py::object  cfg      = json_mod.attr("load")(cfg_file);
            cfg_file.attr("close")();
            elite_weights_path = cfg["training"]["elite_weights_path"]
                                    .cast<std::string>();

            std::cerr << "[DQNBridge] Agent " << agent_id
                      << " initialized. Device: "
                      << agent.attr("device").attr("type").cast<std::string>()
                      << std::endl;
        }
        catch (const py::error_already_set& e) {
            std::cerr << "[DQNBridge] Python error: " << e.what() << std::endl;
            throw;
        }
    }

    ~Impl() {
        s_interpreter_ref_count--;
        if (s_interpreter_ref_count == 0) {
            delete s_interpreter;
            s_interpreter = nullptr;
        }
    }
};

// ---------------------------------------------------------------------------

DQNBridge::DQNBridge(int agent_id,
                     const std::string& config_path,
                     const std::string& module_dir)
    : pImpl(std::make_unique<Impl>(agent_id, config_path, module_dir))
{}

DQNBridge::~DQNBridge() = default;

// ---------------------------------------------------------------------------

int DQNBridge::act(const std::vector<double>& state)
{
    try {
        // Преобразуем double → float (PyTorch использует float32)
        std::vector<float> state_f(state.begin(), state.end());
        py::object result = pImpl->agent.attr("act")(state_f);
        return result.cast<int>();
    }
    catch (const py::error_already_set& e) {
        std::cerr << "[DQNBridge] act() error: " << e.what() << std::endl;
        // Возвращаем действие по умолчанию (positioning) при ошибке
        return 8;
    }
}

// ---------------------------------------------------------------------------

void DQNBridge::pushAndTrain(const std::vector<double>& state,
                              int action,
                              double reward,
                              const std::vector<double>& next_state,
                              bool done,
                              int tau)
{
    try {
        std::vector<float> state_f(state.begin(), state.end());
        std::vector<float> next_f(next_state.begin(), next_state.end());

        // Сохраняем кортеж перехода e^i_t = (s^i_t, o^i_t, R^i_t, s^i_{t+tau}, tau^i_t)
        pImpl->agent.attr("push_transition")(
            state_f, action, (float)reward, next_f, done, tau
        );

        // Один шаг обучения по уравнению Беллмана для SMDP
        pImpl->agent.attr("train")();

        pImpl->episode_reward += reward;
    }
    catch (const py::error_already_set& e) {
        std::cerr << "[DQNBridge] pushAndTrain() error: " << e.what() << std::endl;
    }
}

// ---------------------------------------------------------------------------

void DQNBridge::saveEliteIfBest(
    const std::vector<std::pair<int, double>>& all_rewards)
{
    try {
        // Преобразуем в Python dict: {agent_id: reward}
        py::dict rewards_dict;
        for (const auto& [id, r] : all_rewards) {
            rewards_dict[py::int_(id)] = py::float_(r);
        }
        pImpl->agent.attr("save_elite_if_best")(rewards_dict);
    }
    catch (const py::error_already_set& e) {
        std::cerr << "[DQNBridge] saveEliteIfBest() error: " << e.what() << std::endl;
    }
}

// ---------------------------------------------------------------------------

void DQNBridge::loadEliteWeights()
{
    try {
        pImpl->agent.attr("load_weights")(pImpl->elite_weights_path);
        std::cerr << "[DQNBridge] Agent " << pImpl->agent_id
                  << " loaded elite weights from "
                  << pImpl->elite_weights_path << std::endl;
    }
    catch (const py::error_already_set& e) {
        std::cerr << "[DQNBridge] loadEliteWeights() error: " << e.what() << std::endl;
    }
}

// ---------------------------------------------------------------------------

void DQNBridge::resetEpisode()
{
    try {
        pImpl->agent.attr("reset_episode")();
        pImpl->episode_reward = 0.0;
    }
    catch (const py::error_already_set& e) {
        std::cerr << "[DQNBridge] resetEpisode() error: " << e.what() << std::endl;
    }
}

// ---------------------------------------------------------------------------

double DQNBridge::getEpisodeReward() const
{
    return pImpl->episode_reward;
}

// ---------------------------------------------------------------------------

void DQNBridge::logEpisodeResult(int our_score, int opp_score)
{
    try {
        pImpl->agent.attr("log_episode_result")(
            our_score,
            opp_score,
            pImpl->episode_reward
        );
    }
    catch (const py::error_already_set& e) {
        std::cerr << "[DQNBridge] logEpisodeResult() error: " << e.what() << std::endl;
    }
}