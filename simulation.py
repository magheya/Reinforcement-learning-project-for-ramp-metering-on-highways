import os
import traci
import random
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
import torch
# from torchrl.data import ListStorage, PrioritizedReplayBuffer
# from rl_replay_buffer import PrioritizedReplayBuffer

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
        # Get relevant traffic data from SUMO simulation
        highway_throughput = traci.edge.getLastStepVehicleNumber("highway_entry")
        ramp_queue = traci.edge.getLastStepHaltingNumber("ramp_entry")
        avg_speed = traci.edge.getLastStepMeanSpeed("highway_entry")

        highway_throughput = min(highway_throughput / 100, 1)  # Cap à 100 véhicules
        ramp_queue = min(ramp_queue / 50, 1)  # Cap à 50 véhicules
        avg_speed = avg_speed / 15  # Normalisé à une vitesse maximale de 15 m/s


        # Pondération ajustée
        alpha = 0.6
        beta = -2
        gamma = 0.5
        
        # Calculate the reward components
        reward = (alpha * highway_throughput) + (beta * ramp_queue) + (gamma * avg_speed)

        if traci.trafficlight.getPhaseDuration("ramp_metering_tl") > 5:
            reward += 0.05  # Phase stable
        
        if traci.simulation.getCollidingVehiclesNumber() > 0:
            reward -= 10.0  # Collision détectée
        
        if avg_speed < 2.0:
            reward -= 5.0  # Embouteillage

        return reward


    def close(self):
        traci.close()


# Define DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # self.memory = deque(maxlen=10000)
        self.memory = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 5e-4
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.target_update_freq = 10  # Mise à jour du réseau cible tous les 10 épisodes
        self.episode_count = 0
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=0.5))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # self.memory.append((state, action, reward, next_state, done))
        # Calculate TD error or priority (you may set a default priority for now, e.g., `abs(reward)`)
        td_error = abs(reward)  # Simplified; you can later use more advanced TD error calculation
        self.memory.add(state, action, reward, next_state, done, priority=td_error)
    
    def act(self, state):
        # Epsilon-Greedy Strategy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action (exploration)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values)  # Best action (exploitation)
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # minibatch = random.sample(self.memory, batch_size)
        # for state, action, reward, next_state, done in minibatch:
        #     target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(next_state[np.newaxis, :], verbose=0))
        #     target_f = self.model.predict(state[np.newaxis, :], verbose=0)
        #     target_f[0][action] = target
        #     self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)

        # Sample minibatch with importance sampling weights
        minibatch, indices, weights = self.memory.sample(batch_size)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(next_state[np.newaxis, :], verbose=0))
            target_f = self.model.predict(state[np.newaxis, :], verbose=0)
            target_f[0][action] = target
            
            # Update model and adjust priority
            self.model.fit(state[np.newaxis, :], target_f, sample_weight=np.array([weights[i]]), epochs=1, verbose=0)
            
            # Calculate updated priority and update buffer
            new_priority = abs(target - np.amax(target_f))
            self.memory.update(indices[i], new_priority)


    def update_exploration(self, avg_reward=None):
        self.episode_count += 1

        if self.exploration_strategy == "exponential_decay":
            # Exponential decay of epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        elif self.exploration_strategy == "adaptive_reset":
            # Periodic reset of epsilon based on reward performance
            if self.episode_count % self.reset_period == 0 and avg_reward is not None:
                if avg_reward < 0:  # Poor performance, increase exploration
                    self.epsilon = 0.7
                else:  # Good performance, reduce exploration
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# Training Loop
if __name__ == "__main__":
    env = SumoRampEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    episodes = 100
    batch_size = 32
    max_steps_per_episode = 1000

    rewards_per_episode = []  # Track total reward per episode
    avg_speed_per_episode = []  # Average speed per episode
    queue_length_per_episode = []  # Queue length per episode

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
        avg_speed_per_episode.append(traci.edge.getLastStepMeanSpeed("highway_entry"))
        queue_length_per_episode.append(traci.edge.getLastStepHaltingNumber("ramp_entry"))
        
        agent.replay(batch_size)
        if e % agent.target_update_freq == 0:
            agent.update_target_model()
        agent.update_exploration()

        print(f"Episode: {e+1}, Reward: {total_reward:.2f}, Avg Speed: {np.mean(avg_speed_per_episode):.2f}, "
        f"Queue: {np.mean(queue_length_per_episode):.2f}, Epsilon: {agent.epsilon:.2f}") 
    
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
    
    agent.model.save("dqn_model.h5")
    env.close()
