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
            "sumo",
            "-c", "RL_project.sumocfg"
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
        alpha = 0.4
        beta = -0.6
        gamma = 0.8
        
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
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 1e-03
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.target_update_freq = 10  # Mise à jour du réseau cible tous les 10 épisodes
        self.exploration_strategy = "linear_decay"  # Can also set to "periodic_reset"
        self.reset_period = 50  # For periodic exploration reset
        self.episode_count = 0
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
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
        # Epsilon-Greedy Strategy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action (exploration)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values)  # Best action (exploitation)
    
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
        if self.exploration_strategy == "linear_decay":
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        elif self.exploration_strategy == "periodic_reset" and self.episode_count % self.reset_period == 0:
            self.epsilon = 0.5  # Redémarrage partiel de l'exploration

def run_episode(env, agent, max_steps_per_episode, batch_size):
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

    agent.replay(batch_size)
    if agent.episode_count % agent.target_update_freq == 0:
        agent.update_target_model()
    agent.update_exploration()

    return total_reward

def run_simulation(run, episodes, max_steps_per_episode, batch_size, rewards_per_episode, lock, csv_file):
    env = SumoRampEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for e in range(episodes):
            total_reward = run_episode(env, agent, max_steps_per_episode, batch_size)
            with lock:
                rewards_per_episode[e] += total_reward
            avg_reward = rewards_per_episode[e] / (run + 1)
            writer.writerow([e + 1, avg_reward])
            file.flush()  # Ensure data is written to the file immediately
            print(f"Run: {run+1}, Episode: {e+1}/{episodes}, Reward: {total_reward:.2f}, Average Reward: {avg_reward:.2f}")
    env.close()

def main():
    num_runs = 5
    episodes = 300
    batch_size = 32
    max_steps_per_episode = 1000
    csv_file = 'rewards.csv'

    manager = Manager()
    rewards_per_episode = manager.list([0] * episodes)
    lock = manager.Lock()

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Average Reward'])

    with Pool(processes=num_runs) as pool:
        pool.starmap(run_simulation, [(run, episodes, max_steps_per_episode, batch_size, rewards_per_episode, lock, csv_file) for run in range(num_runs)])

    # Calculate the average rewards across all runs
    avg_rewards = [reward / num_runs for reward in rewards_per_episode]

    # Plot the average rewards
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Across Episodes')
    plt.show()

if __name__ == "__main__":
    main()