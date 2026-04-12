import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_logs(log_path):
    if not os.path.exists(log_path):
        print(f"Файл {log_path} не найден!")
        return

    data = pd.read_csv(log_path)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    window = 50 
    data['Smoothed_Reward'] = data['AvgReward'].rolling(window=window, min_periods=1).mean()
    data['Smoothed_Loss'] = data['Loss'].rolling(window=window, min_periods=1).mean()

    # 1. График Epsilon
    ax1.plot(data['Step'], data['Epsilon'], color='green', linewidth=2)
    ax1.set_title('Изменение ε (Exploration vs Exploitation)', fontsize=14)
    ax1.set_ylabel('Epsilon')
    ax1.grid(True)

    # 2. График Loss
    ax2.plot(data['Step'], data['Loss'], color='red', alpha=0.3, label='Оригинал')
    ax2.plot(data['Step'], data['Smoothed_Loss'], color='darkred', linewidth=2, label='Сглаженный')
    ax2.set_title('Функция потерь (Loss)', fontsize=14)
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # 3. График Вознаграждения
    ax3.plot(data['Step'], data['AvgReward'], color='blue', alpha=0.3, label='Оригинал')
    ax3.plot(data['Step'], data['Smoothed_Reward'], color='darkblue', linewidth=2, label='Сглаженный')
    ax3.set_title('Среднее вознаграждение (Average Reward)', fontsize=14)
    ax3.set_xlabel('Шаги обучения')
    ax3.set_ylabel('Reward')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(log_path), 'training_results.png')
    plt.savefig(save_path)
    print(f"Графики успешно сохранены в: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    log_file = "/Users/laaimak/Desktop/VKR/done_brain/agent_2_steps.csv"
    # log_file = "/Users/laaimak/Desktop/VKR/helios-base/src/logs/agent_2_steps.csv"
    plot_training_logs(log_file)