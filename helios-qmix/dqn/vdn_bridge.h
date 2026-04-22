#pragma once

/**
 * VDNBridge — мост между C++ агентом и VDN системой.
 * Два режима (переключается переменной окружения VDN_INFERENCE=1):
 * Это реализует принцип CTDE:
 *   Centralized Training  → Flask сервер видит всех агентов
 *   Decentralized Execution → каждый агент автономен в игре
 */

#include <string>
#include <vector>
#include <memory>

class VDNBridge {
public:
    enum class Mode { TRAIN, INFERENCE };

    VDNBridge(int agent_id,
              const std::string& config_path,
              const std::string& module_dir);
    ~VDNBridge();

    int reset(const std::vector<double>& state);

    int step(const std::vector<double>& state,
             double reward,
             bool done);

    // Выбор действия без обновления (только inference).
    int selectAction(const std::vector<double>& state);

    Mode   mode()       const { return M_mode;       }
    int    agentId()    const { return M_agent_id;   }
    double epsilon()    const { return M_epsilon;     }
    int    stepsDone()  const { return M_steps_done; }
    bool   isInference() const { return M_mode == Mode::INFERENCE; }

    // Возвращает true если Python-интерпретатор был создан в этом процессе
    static bool isPythonStarted();

private:
    int         M_agent_id;
    Mode        M_mode;
    std::string M_host;
    int         M_port;
    double      M_epsilon    = 1.0;
    int         M_steps_done = 0;

    // TRAIN MODE: HTTP клиент
    std::string httpPost(const std::string& path,
                         const std::string& json_body);
    std::string buildStepJson(const std::vector<double>& state,
                              double reward, bool done) const;
    std::string buildResetJson(const std::vector<double>& state) const;
    int         parseAction(const std::string& json) const;
    void        parseMetadata(const std::string& json);

    // INFERENCE MODE: локальная Q-сеть
    std::string  M_module_dir;
    std::string  M_config_path;
    bool         M_py_initialized = false;

    void initPython();
    int  localSelectAction(const std::vector<double>& state);
};
