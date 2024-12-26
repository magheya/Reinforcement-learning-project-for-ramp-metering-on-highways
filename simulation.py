import os
import traci
import random
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
import csv
from multiprocessing import Pool, Manager


# Define SUMO environment
class SumoRampEnv:
    def __init__(self):
        self.sumoCmd = [
            "sumo-gui",
            "-c", "E:/ESTIN/RLOC_Project/RL_project.sumocfg"
        ]
        self.actions = [0, 1, 2]  # Green, Yellow, Red
        self.state_size = 6  # State dimensions
        self.action_size = len(self.actions)
        self.reset()

    def reset(self):
        try:
            traci.close()
        except Exception:
            pass
        traci.start(self.sumoCmd)
        return self.get_state()
    
    def step(self, action):
        try:
            if action == 0:
                traci.trafficlight.setPhase("ramp_metering_tl", 0)  # Green
            elif action == 1:
                traci.trafficlight.setPhase("ramp_metering_tl", 1)  # Yellow
            else:
                traci.trafficlight.setPhase("ramp_metering_tl", 2)  # Red

            traci.simulationStep()
            next_state = self.get_state()
            reward = self.calculate_reward()
            done = traci.simulation.getMinExpectedNumber() <= 0
            return next_state, reward, done
        except traci.exceptions.TraCIException as e:
            print(f"Error in step function: {e}")
            return np.zeros(self.state_size), -1, True
    
    def get_state(self):
        try:
            highway_density = traci.edge.getLastStepVehicleNumber("highway_entry") / 3   
            ramp_density = traci.edge.getLastStepVehicleNumber("ramp_entry")
            avg_speed = traci.edge.getLastStepMeanSpeed("highway_entry")
            queue_length = traci.edge.getLastStepHaltingNumber("ramp_entry")
            traffic_light_phase = traci.trafficlight.getPhase("ramp_metering_tl")
            phase_duration = traci.trafficlight.getPhaseDuration("ramp_metering_tl")
            
            return np.array([ 
                highway_density, ramp_density, avg_speed, queue_length,
                traffic_light_phase, phase_duration
            ])
        except traci.exceptions.TraCIException as e:
            print(f"Error during state retrieval: {e}")
            return np.zeros(self.state_size)

    def calculate_reward(self):
        throughput = traci.edge.getLastStepVehicleNumber("highway_entry")
        ramp_queue = traci.edge.getLastStepHaltingNumber("ramp_entry")
        avg_speed = traci.edge.getLastStepMeanSpeed("highway_entry")

        alpha = 0.7
        beta = -4
        gamma = 1.5

        reward = (alpha * throughput) + (beta * ramp_queue) + (gamma * avg_speed)

        if avg_speed < 2.0:
            reward -= 2.0  # Penalty for severe congestion

        collisions = traci.simulation.getCollidingVehiclesNumber()
        if collisions > 0:
            reward -= 5.0  # Moderate collision penalty

        return reward

    def close(self):
        traci.close()


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.0005
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.target_update_freq = 10  # Target model update frequency
        self.episode_count = 0

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Softmax policy without temperature scaling
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        exp_q_values = np.exp(q_values)  # Apply softmax without temperature
        probabilities = exp_q_values / np.sum(exp_q_values)  # Normalize to get a probability distribution
        
        # Sample an action based on the probabilities
        action = np.random.choice(self.action_size, p=probabilities)
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(next_state[np.newaxis, :], verbose=0))
            target_f = self.model.predict(state[np.newaxis, :], verbose=0)
            target_f[0][action] = target
            self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)

    def update_exploration(self):
        self.episode_count += 1
        # Decay epsilon for exploration vs exploitation (optional)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

with open('traffic_metrics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Average Reward', 'Average Speed (m/s)', 'Queue Length', 'Congestion Rate'])

# Training Loop
if __name__ == "__main__":
    env = SumoRampEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    episodes = 1000
    batch_size = 32
    max_steps_per_episode = 1000

    rewards_per_episode = []  # Track total reward per episode
    avg_speed_per_episode = []  # Average speed per episode
    queue_length_per_episode = []  # Queue length per episode
    congestion_rate_per_episode = []  # Congestion rate per episode

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0 

        while not done and step_count < max_steps_per_episode:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

        rewards_per_episode.append(total_reward)
        avg_speed = traci.edge.getLastStepMeanSpeed("highway_entry")
        avg_speed_per_episode.append(avg_speed)
        queue_length = traci.edge.getLastStepHaltingNumber("ramp_entry")
        queue_length_per_episode.append(queue_length)
        # Calculate Congestion Rate
        max_queue_length = 50  # Assuming max queue length is 50 vehicles
        congestion_rate = queue_length / max_queue_length
        congestion_rate_per_episode.append(congestion_rate)

        agent.replay(batch_size)
        if e % agent.target_update_freq == 0:
            agent.update_target_model()
        agent.update_exploration()

        print(f"Episode: {e+1}, Reward: {total_reward:.2f}, Avg Speed: {np.mean(avg_speed_per_episode):.2f}, "
              f"Queue: {np.mean(queue_length_per_episode):.2f}, Congestion Rate: {np.mean(congestion_rate_per_episode):.2f}, Epsilon: {agent.epsilon:.2f}")

        #Save metrics to CSV after each episode
        with open('traffic_metrics.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([e+1, total_reward, avg_speed, queue_length, congestion_rate])

    # Plot Total Reward Across Episodes
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_episode, label='Total Reward', color='b')
    plt.title('Total Reward Across Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()

    # Plot Average Speed Across Episodes
    plt.figure(figsize=(12, 6))
    plt.plot(avg_speed_per_episode, label='Average Speed', color='g')
    plt.title('Average Vehicle Speed Across Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Average Speed (m/s)')
    plt.legend()
    plt.show()

    # Plot Queue Length Across Episodes
    plt.figure(figsize=(12, 6))
    plt.plot(queue_length_per_episode, label='Queue Length', color='r')
    plt.title('Average Queue Length Across Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Halted Vehicles')
    plt.legend()
    plt.show()

    # Plot Congestion Rate Across Episodes
    plt.figure(figsize=(12, 6))
    plt.plot(congestion_rate_per_episode, label='Congestion Rate', color='purple')
    plt.title('Congestion Rate Across Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Congestion Rate')
    plt.legend()
    plt.show()

    agent.model.save("dqn_model.h5")
    env.close()