import numpy as np
import gym
import setuptools.dist
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size

        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_decay = 0.005

        self.batch_size = 32
        self.memory_buffer = list()
        self.max_memory_buffer = 2000

        self.model = Sequential([
            Input(shape=(state_size,)),
            Dense(units=24, activation='relu'),
            Dense(units=24, activation='relu'),
            Dense(units=action_size, activation='linear')
        ])

        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))

    def compute_action(self, current_state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(range(self.n_actions))
        else:
            q_values = self.model.predict(current_state)[0]
            return np.argmax(q_values)
        
    def update_epsilon(self):
        self.epsilon = self.epsilon * np.exp(-self.epsilon_decay)

    def store_episode(self, current_state, action, reward, next_state, done):
        self.memory_buffer.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })

        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)

    def train(self):
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]

        for experience in batch_sample:
            q_current_state = self.model.predict(experience["current_state"])[0]
            
            #Calculate Q by Bellman
            if not experience["done"]:
                q_target = experience["reward"] + self.gamma * np.max(self.model.predict(experience["next_state"])[0])
            else:
                q_target = experience["reward"]

            q_current_state[experience["action"]] = q_target

            # treinar o modelo
            self.model.fit(experience["current_state"], np.array([q_current_state]), verbose=0)
            
    def save_model_weights(self):
        self.model.save_weights('./models/dqn-model.weights.h5')

    def load_model_weights(self):
        self.model.load_weights('./models/dqn-model.weights.h5')

def run_training_routine():
    # Treinar o modelo
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    print(state_size)
    action_size = env.action_space.n

    n_episodes = 600
    max_iteration_ep = 500

    agent = DQNAgent(state_size, action_size)
    total_steps = 0

    for episode in range(n_episodes):
        print(f"Episódio {episode}...")
        current_state = env.reset()
        current_state = np.array([current_state[0]])

        for iteration in range(max_iteration_ep):
            total_steps += 1

            action = agent.compute_action(current_state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array([next_state])

            agent.store_episode(current_state, action, reward, next_state, done)

            if done:
                agent.update_epsilon()
                break

            current_state = next_state

        if total_steps >= agent.batch_size and total_steps % 5 == 0:
            agent.train()

    agent.save_model_weights()

def run_sim_routine():
    env = gym.make('CartPole-v1', render_mode="human")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.load_model_weights()
    rewards = 0
    steps = 0
    done = False
    state = env.reset()
    state = np.array([state[0]])

    while not done:
        action = agent.compute_action(state)
        state, reward, done, _, _ = env.step(action)
        state = np.array([state])            
        steps += 1
        rewards += reward

    print(rewards)
    env.close()

if __name__ == "__main__":
    run_training_routine()