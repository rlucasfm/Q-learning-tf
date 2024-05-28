import gym.logger
import numpy as np
import gym
import setuptools.dist
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import time
from timeit import default_timer as timer

class DDQNAgent:
    def __init__(self, state_size, action_size):
        # Hyperparameters and misc
        self.n_actions = action_size
        self.lr = 0.005
        self.gamma = 0.85
        self.tau = 0.05
        self.epsilon = 1.0
        self.epsilon_decay = 0.05

        # Experience Replay
        self.memory_buffer_size = int(1e5)
        self.memory_buffer_pointer = 0
        self.memory_buffer_current_state = np.empty(shape=(self.memory_buffer_size, state_size))
        self.memory_buffer_next_state = np.empty(shape=(self.memory_buffer_size, state_size))
        self.memory_buffer_action = np.empty(shape=(self.memory_buffer_size,))
        self.memory_buffer_reward = np.empty(shape=(self.memory_buffer_size,))
        self.memory_buffer_done = np.empty(shape=(self.memory_buffer_size,))

        # Neural Networks 
        self.q_model = self.build_model(state_size, action_size)
        self.q_target_model = self.build_model(state_size, action_size)

    def build_model(self, state_size, action_size):
        model = Sequential([
            Input(shape=(state_size,)),
            Dense(units=64, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
        return model
    
    def compute_action(self, current_state):
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.choice(range(self.n_actions))
        else:
            q_values = self.q_model.predict(current_state)[0]
            return np.argmax(q_values)
        
    def store_episode(self, current_state, action, reward, next_state, done):
        self.memory_buffer_current_state[self.memory_buffer_pointer] = current_state
        self.memory_buffer_next_state[self.memory_buffer_pointer] = next_state
        self.memory_buffer_action[self.memory_buffer_pointer] = action
        self.memory_buffer_reward[self.memory_buffer_pointer] = reward
        self.memory_buffer_done[self.memory_buffer_pointer] = done
        
        self.memory_buffer_pointer += 1

        if self.memory_buffer_pointer >= self.memory_buffer_size:
            self.memory_buffer_pointer = 0

    def update_epsilon(self):
        self.epsilon = self.epsilon_decay * self.epsilon

    def train(self, batch_size):
        batch_indices = np.random.choice(self.memory_buffer_current_state.shape[0], batch_size)

        current_states = self.memory_buffer_current_state[batch_indices]
        rewards = self.memory_buffer_reward[batch_indices]
        done = self.memory_buffer_done[batch_indices]

        q_current_states = self.q_model.predict(current_states)
        q_current_states_target = self.q_target_model.predict(current_states)

        # q_targets = rewards + (1 - done) * (self.gamma * np.amax(q_current_states_target, axis=1))
        # action_indices = np.argmax(q_current_states_target, axis=1)

        q_targets = np.empty(len(q_current_states_target))
        for i in range(len(q_current_states_target)):
            if done[i]:
                q_targets[i] = rewards[i] 
            else:
                q_targets[i] = rewards[i] + self.gamma * np.max(q_current_states_target[i])

            q_current_states[i][np.argmax(q_current_states_target[i])] = q_targets[i]

        self.q_model.fit(current_states, q_current_states, batch_size=batch_size, verbose=0, epochs=10)

    def update_q_target_network(self):
        model_weights = [w * self.tau for w in self.q_model.get_weights()]
        target_weights = [w * (1 - self.tau) for w in self.q_target_model.get_weights()]
        
        combined_weights = [mw + tw for mw, tw in zip(model_weights, target_weights)]
        self.q_target_model.set_weights(combined_weights)

    def save_model_weights(self):
        self.q_model.save_weights('./models/double-dqn-model.weights.h5')

    def load_model_weights(self):
        self.q_model.load_weights('./models/double-dqn-model.weights.h5')
        self.q_target_model.load_weights('./models/double-dqn-model.weights.h5')

def run_training_routine():
    start_time = timer()
    env = gym.make("CartPole-v1")
    success_threshold = 180
    successfull_episodes_threshold = 20
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    n_episodes = 500
    max_iterations_ep = 400
    batch_size = 64
    q_network_update_freq = 4
    reward_history = []

    agent = DDQNAgent(state_size, action_size)
    total_steps = 0
    successfull_episodes = 0

    for episode in range(n_episodes):
        episode_start = timer()

        print(f"Episódio {episode}")
        current_state = env.reset()
        current_state = np.array([current_state[0]])

        rewards = 0

        for step in range(max_iterations_ep):
            total_steps += 1

            action = agent.compute_action(current_state)        
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array([next_state])

            rewards = rewards + reward
            agent.store_episode(current_state, action, reward, next_state, done)
            
            if done:
                if rewards >= success_threshold:
                    successfull_episodes += 1
                agent.update_epsilon()
                break

            current_state = next_state
        
        print(" rewards: ", rewards)
        reward_history.append(rewards)

        if total_steps >= batch_size:
            if total_steps % q_network_update_freq == 0:
                agent.train(batch_size=batch_size)
                agent.update_q_target_network()                

        print(f"Successfull episodes: {successfull_episodes}")

        episode_end = timer()
        print(f"Episode elapsed time: {episode_end - episode_start}")

        if successfull_episodes >= successfull_episodes_threshold:
            print("Successfull episodes threshold reached.")
            print(f"Total Elapsed time: {episode_end - start_time}")
            agent.save_model_weights()
            return reward_history

    end_time = timer()
    print(f"Total Elapsed time: {end_time - start_time}")
    agent.save_model_weights()
    return reward_history

def run_sim_routine():
    env = gym.make('CartPole-v1', render_mode="human")
    agent = DDQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.load_model_weights()
    rewards = 0
    done = False
    state = env.reset()
    state = np.array([state[0]])

    while not done:
        action = agent.compute_action(state)
        state, reward, done, _, _ = env.step(action)
        rewards += reward
        time.sleep(0.05)

    print(f"Rewards: {rewards}.")
    env.close()

if __name__ == "__main__":
    reward_history = run_training_routine()
    fig, ax = plt.subplots()
    ax.plot(reward_history)
    ax.set(xlabel="Iteração", ylabel="Recompensa do episódio")
    ax.grid()
    plt.show()

    # for i in range(5):
    #     run_sim_routine()