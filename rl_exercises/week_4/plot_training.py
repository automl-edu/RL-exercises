import subprocess
import re
import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt

CONFIGS = [
    {"name": "batch_size_32", "overrides": "agent.batch_size=32", "config": "dqn"},
    {"name": "batch_size_64", "overrides": "agent.batch_size=64", "config": "dqn"},
    {"name": "batch_size_128", "overrides": "agent.batch_size=128", "config": "dqn"},
    {"name": "buffer_5000", "overrides": "agent.buffer_capacity=5000", "config": "dqn"},
    {"name": "buffer_10000", "overrides": "agent.buffer_capacity=10000", "config": "dqn"},
    {"name": "buffer_20000", "overrides": "agent.buffer_capacity=20000", "config": "dqn"},
    {"name": "net_small", "overrides": "", "config": "dqn_small"},
    {"name": "net_medium", "overrides": "", "config": "dqn_medium"},
    {"name": "net_large", "overrides": "", "config": "dqn_large"},
]

RESULTS_DIR = Path(__file__).parent / "training_results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_training(config_name, overrides, config_name_override):
    """Execute DQN training with specified config"""
    print(f"\nRunning: {config_name}")
    cmd = f"python -m rl_exercises.week_4.dqn --config-name {config_name_override}"
    if overrides:
        cmd += f" {overrides}"
    
    output_file = RESULTS_DIR / f"{config_name}_output.txt"
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    with open(output_file, "w", encoding='utf-8') as f:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd="d:\\Python\\Reinforcement Learning Uni\\RL-exercises",
            env=env
        )
        f.write(result.stdout)
        f.write(result.stderr)
    
    print(f"Output saved to {output_file}")
    return output_file


def parse_output(output_file):
    """Extract frames and rewards from training output file"""
    frames = []
    rewards = []
    
    with open(output_file, "r") as f:
        content = f.read()
    
    pattern = r"Frame (\d+), AvgReward\(10\): ([\d.]+)"
    matches = re.findall(pattern, content)
    
    for frame_str, reward_str in matches:
        frames.append(int(frame_str))
        rewards.append(float(reward_str))
    
    print(f"Parsed {len(frames)} data points")
    return frames, rewards


def save_to_csv(config_name, frames, rewards):
    """Save frames and rewards to CSV file"""
    csv_file = RESULTS_DIR / f"{config_name}_results.csv"
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "AvgReward"])
        for frame, reward in zip(frames, rewards):
            writer.writerow([frame, reward])
    
    print(f"Results saved to {csv_file}")


def plot_results(all_results):
    """Generate comparison plot of all training curves"""
    plt.figure(figsize=(12, 7))
    
    for config_name, (frames, rewards) in all_results.items():
        if frames and rewards:
            plt.plot(frames, rewards, marker="o", markersize=3, label=config_name, linewidth=2)
    
    plt.xlabel("Frames", fontsize=12)
    plt.ylabel("Average Reward (last 10 episodes)", fontsize=12)
    plt.title("DQN Training Curves - Architecture, Batch Size, and Buffer Size", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = RESULTS_DIR / "training_curves.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to {plot_file}")
    
    plt.show()


def main():
    """Run all configurations and generate comparison results"""
    
    all_results = {}
    
    for config in CONFIGS:
        config_name = config["name"]
        config_override = config["config"]
        output_file = run_training(config_name, config["overrides"], config_override)
        frames, rewards = parse_output(output_file)
        
        if frames and rewards:
            save_to_csv(config_name, frames, rewards)
            all_results[config_name] = (frames, rewards)
    
    if all_results:
        plot_results(all_results)
        print("Analysis complete!")


if __name__ == "__main__":
    main()
