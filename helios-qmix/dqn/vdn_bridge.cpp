#include "vdn_bridge.h"

#include <curl/curl.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace py = pybind11;

// Глобальный Python интерпретатор (для inference mode)
static py::scoped_interpreter* g_interpreter = nullptr;
static py::object               g_agent_obj;   // экземпляр DQNAgent

bool VDNBridge::isPythonStarted() { return g_interpreter != nullptr; }

static size_t writeCallback(char* ptr, size_t size, size_t nmemb, std::string* data)
{
    data->append(ptr, size * nmemb);
    return size * nmemb;
}

VDNBridge::VDNBridge(int agent_id,
                     const std::string& config_path,
                     const std::string& module_dir)
    : M_agent_id(agent_id)
    , M_host("localhost")
    , M_port(6100)
    , M_module_dir(module_dir)
    , M_config_path(config_path)
{
    // Определяем режим из переменной окружения
    const char* inf_env = std::getenv("VDN_INFERENCE");
    M_mode = (inf_env && std::string(inf_env) == "1")
             ? Mode::INFERENCE
             : Mode::TRAIN;

    if (M_mode == Mode::TRAIN) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        std::cerr << "[VDNBridge] TRAIN mode, agent=" << M_agent_id
                  << " server=" << M_host << ":" << M_port << std::endl;
    } else {
        std::cerr << "[VDNBridge] INFERENCE mode, agent=" << M_agent_id
                  << " (локальная Q-сеть)" << std::endl;
        initPython();
    }
}

VDNBridge::~VDNBridge()
{
    if (M_mode == Mode::TRAIN) {
        curl_global_cleanup();
    } else {

        if (g_interpreter) {
            std::fflush(nullptr);
            std::_Exit(0);
        }
    }
}

int VDNBridge::reset(const std::vector<double>& state)
{
    if (M_mode == Mode::INFERENCE) {
        return localSelectAction(state);
    }

    try {
        std::string resp = httpPost("/reset", buildResetJson(state));
        parseMetadata(resp);
        return parseAction(resp);
    } catch (const std::exception& e) {
        std::cerr << "[VDNBridge] reset error: " << e.what() << std::endl;
        return 6;
    }
}

int VDNBridge::step(const std::vector<double>& state,
                    double reward, bool done)
{
    if (M_mode == Mode::INFERENCE) {
        return localSelectAction(state);
    }

    // TRAIN: блокируется пока все 10 агентов не отчитаются
    try {
        std::string resp = httpPost("/step", buildStepJson(state, reward, done));
        parseMetadata(resp);
        return parseAction(resp);
    } catch (const std::exception& e) {
        std::cerr << "[VDNBridge] step error: " << e.what() << std::endl;
        return 6;
    }
}

int VDNBridge::selectAction(const std::vector<double>& state)
{
    if (M_mode == Mode::INFERENCE) {
        return localSelectAction(state);
    }
    return step(state, 0.0, false);
}

// INFERENCE MODE: локальная Q-сеть

void VDNBridge::initPython()
{
    if (M_py_initialized) return;

    try {
        if (!g_interpreter) {
            g_interpreter = new py::scoped_interpreter();
        }

        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, "/Users/laaimak/Desktop/VKR/.venv314/lib/python3.14/site-packages");
        sys.attr("path").attr("insert")(0, M_module_dir);

        // Загружаем SMDPAgent с VDN весами (индивидуальная Q-сеть)
        py::module_ agent_mod = py::module_::import("agent");
        g_agent_obj = agent_mod.attr("SMDPAgent")(
            M_agent_id,
            M_config_path
        );

        // Загружаем VDN веса для этого агента.
        const char* weights_dir_env = std::getenv("VDN_WEIGHTS_DIR");
        std::string weights_path = weights_dir_env
            ? (std::string(weights_dir_env) + "/vdn_agent_" + std::to_string(M_agent_id) + ".pth")
            : (M_module_dir + "/../helios-qmix/src/logs/vdn_agent_" + std::to_string(M_agent_id) + ".pth");

        py::object load_fn = g_agent_obj.attr("load_weights");
        if (py::module_::import("os").attr("path").attr("exists")(weights_path).cast<bool>()) {
            load_fn(weights_path);
            std::cerr << "[VDNBridge] Загружены VDN веса: vdn_agent_"
                      << M_agent_id << ".pth" << std::endl;
        } else {
            std::cerr << "[VDNBridge] VDN веса не найдены, используем IQL checkpoint" << std::endl;
        }

        M_py_initialized = true;
    } catch (const py::error_already_set& e) {
        std::cerr << "[VDNBridge] Python init error: " << e.what() << std::endl;
    }
}

int VDNBridge::localSelectAction(const std::vector<double>& state)
{
    if (!M_py_initialized) {
        initPython();
    }

    try {
        py::gil_scoped_acquire acquire;
        // epsilon=0 при инференсе — всегда жадная стратегия
        g_agent_obj.attr("epsilon") = py::float_(0.0);
        std::vector<float> state_f(state.begin(), state.end());
        py::object action = g_agent_obj.attr("act")(state_f);
        return action.cast<int>();
    } catch (const py::error_already_set& e) {
        std::cerr << "[VDNBridge] localSelectAction error: " << e.what() << std::endl;
        return 6;
    }
}

// HTTP POST (train mode)

std::string VDNBridge::httpPost(const std::string& path,
                                const std::string& json_body)
{
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    std::string url = "http://" + M_host + ":" + std::to_string(M_port) + path;
    std::string response;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL,          url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST,          1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,    json_body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,    headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,     &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,       10L);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("curl: ") + curl_easy_strerror(res));
    }
    return response;
}

std::string VDNBridge::buildResetJson(const std::vector<double>& state) const
{
    std::ostringstream oss;
    oss << "{\"agent_id\":" << M_agent_id << ",\"state\":[";
    for (size_t i = 0; i < state.size(); ++i) {
        oss << state[i];
        if (i + 1 < state.size()) oss << ",";
    }
    oss << "]}";
    return oss.str();
}

std::string VDNBridge::buildStepJson(const std::vector<double>& state,
                                     double reward, bool done) const
{
    std::ostringstream oss;
    oss << "{\"agent_id\":"  << M_agent_id
        << ",\"reward\":"    << reward
        << ",\"done\":"      << (done ? "true" : "false")
        << ",\"state\":[";
    for (size_t i = 0; i < state.size(); ++i) {
        oss << state[i];
        if (i + 1 < state.size()) oss << ",";
    }
    oss << "]}";
    return oss.str();
}

int VDNBridge::parseAction(const std::string& json) const
{
    auto pos = json.find("\"action\"");
    if (pos == std::string::npos) return 6;
    pos = json.find(":", pos);
    if (pos == std::string::npos) return 6;
    ++pos;
    while (pos < json.size() && json[pos] == ' ') ++pos;
    return std::stoi(json.substr(pos));
}

void VDNBridge::parseMetadata(const std::string& json)
{
    auto parseDouble = [&](const std::string& key) -> double {
        auto pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return -1.0;
        pos = json.find(":", pos);
        if (pos == std::string::npos) return -1.0;
        ++pos;
        while (pos < json.size() && json[pos] == ' ') ++pos;
        try { return std::stod(json.substr(pos)); }
        catch (...) { return -1.0; }
    };

    double eps = parseDouble("epsilon");
    double stp = parseDouble("steps");
    if (eps >= 0.0) M_epsilon    = eps;
    if (stp >= 0.0) M_steps_done = static_cast<int>(stp);
}
