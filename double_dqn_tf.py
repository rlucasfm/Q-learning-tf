import numpy as np
import gym
import setuptools.dist
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        self.lr = 0.05
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.005
        self.memory_buffer = list()
        self.memory_buffer_size = 2000
        self.q_model = self.build_model(state_size, action_size)
        self.q_target_model = self.build_model(state_size, action_size)

    def build_model(self, state_size, action_size):
        model = Sequential([
            Input(shape=(state_size,)),
            Dense(units=24, activation='relu'),
            Dense(units=24, activation='relu'),
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

    def update_q_target_network(self):
        self.q_target_model.set_weights(self.q_model.get_weights())

    def save_model_weights(self):
        self.q_model.save_weights('./models/double-dqn-model.weights.h5')

    def load_model_weights(self):
        self.q_model.load_weights('./models/double-dqn-model.weights.h5')
        self.q_target_model.load_weights('./models/double-dqn-model.weights.h5')

def run_training_routine():
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    n_episodes = 1000
    max_iterations_ep = 400
    batch_size = 64
    q_target_update_freq = 10

    agent = DDQNAgent(state_size, action_size)
    total_steps = 0
    n_training = 0

    for episode in range(n_episodes):
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
                agent.update_epsilon()
                break

            current_state = next_state
        
        print(" rewards: ", rewards)

        if total_steps >= batch_size:
            agent.train(batch_size=batch_size)
            n_training = n_training + 1
            if n_training % q_target_update_freq:
                agent.update_q_target_network()

    agent.save_model_weights()

def run_sim_routine():
    env = gym.make('CartPole-v1', render_mode="human")
    agent = DDQNAgent(env.observation_space.shape[0], env.action_space.n)
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