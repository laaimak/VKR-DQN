import os
import json

BASE_PATHS = [
    "/Users/laaimak/Desktop/VKR/helios-base/src",
    "/Users/laaimak/Desktop/VKR/helios-original/src"
]

FOLDERS = ["formations-dt", "formations-keeper", "formations-taker"]

AGENT_A = "2"
AGENT_B = "11"

def swap_roles():
    count = 0
    for base_path in BASE_PATHS:
        for folder in FOLDERS:
            folder_path = os.path.join(base_path, folder)
            if not os.path.exists(folder_path):
                continue

            for filename in os.listdir(folder_path):
                if not filename.endswith(".conf"):
                    continue
                    
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    if "role" in data:
                        idx_A = next((i for i, r in enumerate(data["role"]) if str(r.get("number")) == AGENT_A), None)
                        idx_B = next((i for i, r in enumerate(data["role"]) if str(r.get("number")) == AGENT_B), None)
                        if idx_A is not None and idx_B is not None:
                            data["role"][idx_A]["number"] = int(AGENT_B)
                            data["role"][idx_B]["number"] = int(AGENT_A)
                    
                    if "data" in data:
                        for entry in data["data"]:
                            if AGENT_A in entry and AGENT_B in entry:
                                entry[AGENT_A], entry[AGENT_B] = entry[AGENT_B], entry[AGENT_A]
                                
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    count += 1
                except Exception as e:
                    print(f"[-] Ошибка в файле {filename}: {e}")
                    
    print(f"Готово! Обработано файлов: {count}.")
    print(f"Теперь Агент №{AGENT_A} играет на позиции Агента №{AGENT_B}!")

if __name__ == "__main__":
    swap_roles()