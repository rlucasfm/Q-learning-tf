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
        self.n_actions = action_size
        self.lr = 0.005
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.005
        self.memory_buffer = list()
        self.memory_buffer_size = 2000
        self.q_model = self.build_model(state_size, action_size)
        self.q_target_model = self.build_model(state_size, action_size)

    def build_model(self, state_size, action_size):
        model = Sequential([
            Input(shape=(state_size,)),
            Dense(units=48, activation='relu'),
            Dense(units=48, activation='relu'),
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
        self.memory_buffer.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })

        if len(self.memory_buffer) >= self.memory_buffer_size:
            self.memory_buffer.pop(0)

    def update_epsilon(self):
        self.epsilon = self.epsilon_decay * self.epsilon

    def train(self, batch_size):
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:batch_size]

        for experience in batch_sample:
            q_current_state = self.q_model.predict(experience["current_state"])[0]

            if experience["done"]:
                q_target = experience["reward"]
            else:
                q_target = experience["reward"] + (self.gamma * np.max(self.q_target_model.predict(experience["next_state"])[0]))

            q_current_state[experience["action"]] = q_target
            self.q_model.fit(experience["current_state"], np.array([q_current_state]), verbose=0)

    def train_v2(self, batch_size):
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:batch_size]
        current_state_arr = []
        q_current_state_arr = []

        for experience in batch_sample:
            q_current_state = self.q_model.predict(experience["current_state"])[0]

            if experience["done"]:
                q_target = experience["reward"]
            else:
                q_target = experience["reward"] + (self.gamma * np.max(self.q_target_model.predict(experience["next_state"])[0]))

            q_current_state[experience["action"]] = q_target
            current_state_arr.append(experience["current_state"])
            q_current_state_arr.append(np.array([q_current_state]))

        self.q_model.fit(current_state_arr, q_current_state_arr, shuffle=True, epochs=1, verbose='auto')

    def update_q_target_network(self):
        self.q_target_model.set_weights(self.q_model.get_weights())

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
    n_episodes = 400
    max_iterations_ep = 200
    batch_size = 128
    q_target_update_freq = 10

    agent = DDQNAgent(state_size, action_size)
    total_steps = 0
    n_training = 0
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

        if total_steps >= batch_size:
            agent.train(batch_size=batch_size)
            n_training = n_training + 1
            if n_training % q_target_update_freq:
                agent.update_q_target_network()

        if successfull_episodes >= successfull_episodes_threshold:
            end_time = timer()
            print("Successfull episodes threshold reached.")
            print(f"Total Elapsed time: {end_time - start_time}")
            return agent.save_model_weights()

        episode_end = timer()
        print(f"Episode elapsed time: {episode_end - episode_start}")

    end_time = timer()
    print(f"Total Elapsed time: {end_time - start_time}")
    agent.save_model_weights()

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
    run_training_routine()

    # for i in range(5):
    #     run_sim_routine()