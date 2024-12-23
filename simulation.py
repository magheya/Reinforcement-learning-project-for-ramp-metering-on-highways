import os
import traci
import random
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

# Define SUMO environment
class SumoRampEnv:
    def __init__(self):
        self.sumoCmd = [
            "sumo",
            "-c", "E:/ESTIN/RLOC_Project/RL_project.sumocfg"
        ]
        self.actions = [0, 1, 2]  # Green, Yellow, Red
        self.state_size = 6  # State dimensions
        self.action_size = len(self.actions)
        traci.start(self.sumoCmd)

    def reset(self):
        traci.close()
        traci.start(self.sumoCmd)
        return self.get_state()
    
    def step(self, action):
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
        highway_congestion = traci.edge.getLastStepHaltingNumber("highway_entry")
        ramp_queue = traci.edge.getLastStepHaltingNumber("ramp_entry")
        avg_speed = traci.edge.getLastStepMeanSpeed("highway_entry")
        
        alpha = 0.5
        beta = 0.5
        gamma = 1.0
        
        reward = -(alpha * ramp_queue) - (beta * highway_congestion) + (gamma * avg_speed)
        
        phase_switch_penalty = -0.1 if traci.trafficlight.getPhase("ramp_metering_tl") == 1 else 0.0
        
        return reward + phase_switch_penalty

    def close(self):
        traci.close()


# Define DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state[np.newaxis, :]))
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(next_state[np.newaxis, :]))
            target_f = self.model.predict(state[np.newaxis, :])
            target_f[0][action] = target
            self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if random.random() < 0.1:
            self.update_target_model()


# Train DQN Agent
if __name__ == "__main__":
    env = SumoRampEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    episodes = 1000
    batch_size = 32
    max_steps_per_episode = 1000  # Define the maximum number of steps per episode

    # Metrics for evaluation
    rewards_per_episode = []  # Track total reward per episode
    avg_speed_per_episode = []  # Average speed per episode
    queue_length_per_episode = []  # Queue length per episode

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0  # Initialize step counter for the episode
        
        while not done and step_count < max_steps_per_episode:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1  # Increment the step counter
        
        rewards_per_episode.append(total_reward)
        avg_speed_per_episode.append(traci.edge.getLastStepMeanSpeed("highway_entry"))
        queue_length_per_episode.append(traci.edge.getLastStepHaltingNumber("ramp_entry"))
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        print(f"Episode: {e+1}, Reward: {total_reward}, Avg Speed: {avg_speed_per_episode}, Queue: {queue_length_per_episode}")

    # Plot metrics
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

    agent.model.save("dqn_model_final.h5")
    env.close()
