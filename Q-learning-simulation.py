import os
import traci
import numpy as np
import random
import matplotlib.pyplot as plt

# Environment Parameters
SUMO_CMD = ["sumo", "-c", "RL_project.sumocfg"]  # Path to your SUMO config file
STATE_SPACE = 5  # Traffic density states
ACTION_SPACE = 3  # Green (0), Yellow (1), Red (2)
EPISODES = 1000
GAMMA = 0.99  # Discount factor
ALPHA = 0.0005  # Learning rate
EPSILON = 0.1  # Exploration-exploitation tradeoff

# Initialize Q-Table
q_table = np.zeros((STATE_SPACE, ACTION_SPACE))

# Reward Function with Multiple Metrics
def calculate_reward():
    waiting_time = sum(traci.edge.getWaitingTime(edge) for edge in ["ramp_entry"])
    flow = traci.edge.getLastStepVehicleNumber("highway_entry")
    avg_speed = traci.edge.getLastStepMeanSpeed("highway_entry")  # Average speed on the highway
    ramp_queue = traci.edge.getLastStepVehicleNumber("ramp_entry")  # Queue length on the ramp
    
    # Reward combines multiple factors
    reward = (-waiting_time * 0.5) + (flow * 1.0) + (avg_speed * 0.3) - (ramp_queue * 0.2)
    return reward

# State Representation
def get_state():
    density = traci.edge.getLastStepOccupancy("highway_entry")
    if density < 0.2:
        return 0
    elif density < 0.4:
        return 1
    elif density < 0.6:
        return 2
    elif density < 0.8:
        return 3
    else:
        return 4

# Action Execution with Yellow Light
def take_action(action):
    if action == 0:
        traci.trafficlight.setPhase("ramp_metering_tl", 0)  # Green Light
    elif action == 1:
        traci.trafficlight.setPhase("ramp_metering_tl", 1)  # Yellow Light
    elif action == 2:
        traci.trafficlight.setPhase("ramp_metering_tl", 2)  # Red Light

# Training Loop
reward_history = []
avg_speed_history = []
waiting_time_history = []

for episode in range(EPISODES):
    traci.start(SUMO_CMD)
    total_reward = 0
    total_waiting_time = 0
    total_avg_speed = 0

    for step in range(1000):  # Simulation steps per episode
        state = get_state()
        if random.uniform(0, 1) < EPSILON:
            action = random.choice(range(ACTION_SPACE))  # Exploration
        else:
            action = np.argmax(q_table[state])  # Exploitation

        take_action(action)
        traci.simulationStep()

        reward = calculate_reward()
        next_state = get_state()
        total_reward += reward
        
        # Collect metrics
        total_waiting_time += sum(traci.edge.getWaitingTime(edge) for edge in ["ramp_entry"])
        total_avg_speed += traci.edge.getLastStepMeanSpeed("highway_entry")

        # Q-Learning Update
        best_next_action = np.argmax(q_table[next_state])
        q_table[state, action] += ALPHA * (reward + GAMMA * q_table[next_state, best_next_action] - q_table[state, action])

    reward_history.append(total_reward)
    avg_speed_history.append(total_avg_speed / 1000)  # Average per step
    waiting_time_history.append(total_waiting_time / 1000)  # Average per step
    
    traci.close()
    print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}, Avg Speed: {avg_speed_history[-1]:.2f}, Avg Waiting Time: {waiting_time_history[-1]:.2f}")

# Plot Results
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot Total Rewards
axs[0].plot(reward_history)
axs[0].set_title('Total Reward per Episode')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Total Reward')

# Plot Average Speed
axs[1].plot(avg_speed_history, color='green')
axs[1].set_title('Average Speed per Episode')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Average Speed (m/s)')

# Plot Average Waiting Time
axs[2].plot(waiting_time_history, color='red')
axs[2].set_title('Average Waiting Time per Episode')
axs[2].set_xlabel('Episode')
axs[2].set_ylabel('Average Waiting Time (s)')

plt.tight_layout()
plt.show()

# Save Q-Table
np.save("q_table.npy", q_table)