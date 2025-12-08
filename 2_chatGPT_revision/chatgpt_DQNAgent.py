# DQNAgent.py
import os
import datetime
from random import random, randint

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

import chatgpt_helpers as helpers
from chatgpt_replay_buffer import ReplayBuffer
from chatgpt_DQN import DQN


class DQNAgent:
    def __init__(
        self,
        device='cpu',
        rom='v0',
        stagesList=['1-1'],
        buttonList='rightOnly',
        randomLevel=True,
        buffer_capacity=50_000,
        batch_size=32,
        gamma=0.99,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=200_000,
        train_freq=4,
    ):
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_freq = train_freq

        # Env
        self.env, self.actionSpace = helpers.initializeEnvironment(
            rom=rom,
            randomLevel=randomLevel,
            stagesList=stagesList,
            buttonList=buttonList,
        )
        self.num_actions = len(self.actionSpace)

        # Observation geometry (post-preprocessing)
        self.ADJ_FRAME_HEIGHT = 100
        self.ADJ_FRAME_WIDTH = 100
        self.obs_shape = (3, self.ADJ_FRAME_HEIGHT, self.ADJ_FRAME_WIDTH)

        # Q-network
        self.Q = DQN(self.obs_shape, self.num_actions).to(self.device)
        self.optimizer = Adam(self.Q.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, self.obs_shape, device=device)

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

    def epsilon(self):
        """Linearly decay epsilon from start to end over epsilon_decay_steps."""
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state_tensor):
        """
        Epsilon-greedy action selection.
        state_tensor: (1, C, H, W) on self.device
        """
        if random() < self.epsilon():
            return randint(0, self.num_actions - 1)
        with torch.no_grad():
            q_values = self.Q(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        return action

    def train_step(self):
        """One gradient step on a batch sampled from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None  # not enough data yet

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Q(s,a)
        q_values = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # max_a' Q(s', a')
        with torch.no_grad():
            next_q_values = self.Q(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_episode(
        self,
        seed=None,
        max_steps=20_000,
        print_status=False,
        print_freq=1_000,
        save_images=False,
        save_image_freq=30,
    ):
        """Run one episode, return cumulative reward and last info dict."""
        cumulative_reward = 0.0

        # Optionally set up image folders
        if save_images:
            timeFolder = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            rawDir = f'./stateSequences/{timeFolder}/raw'
            preproDir = f'./stateSequences/{timeFolder}/preprocessed'
            os.makedirs(rawDir, exist_ok=True)
            os.makedirs(preproDir, exist_ok=True)

        state, info = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        frame = helpers.preprocessFrame(state)
        state_t = helpers.tensorify(frame).to(self.device)

        loss_val = None

        for step in range(max_steps):
            self.total_steps += 1

            # Select action
            action_idx = self.select_action(state_t)
            action_text = self.actionSpace[action_idx]

            # Step the env
            next_state, reward, terminated, truncated, info = self.env.step(action_idx)

            # Preprocess next frame
            next_frame = helpers.preprocessFrame(next_state)
            next_state_t = helpers.tensorify(next_frame).to(self.device)

            done = terminated or truncated
            cumulative_reward += reward

            # Store transition
            self.replay_buffer.store(
                state_t, action_idx, reward, next_state_t, done
            )

            # Train every self.train_freq steps
            if self.total_steps % self.train_freq == 0:
                loss_val = self.train_step()

            # Save images if requested
            if save_images and step % save_image_freq == 0:
                rawRectangle = [0, 0, 70, 50]
                preproRectangle = [0, 0, 70, 50]
                helpers.saveDiagnosticImage(rawDir, state, step, action_text, info['x_pos'], info['y_pos'], rawRectangle)
                helpers.saveDiagnosticImage(preproDir, frame * 255.0, step, action_text, info['x_pos'], info['y_pos'], preproRectangle)
                
            if done:
                break

            state_t = next_state_t

        return {
            "cumulativeReward": cumulative_reward,
            "info": info,
            "loss": loss_val,
        }

    def run_episodes(self, num_episodes, save_models_every=1000):
        """Loop over episodes, periodically save models and log rewards."""
        resultFolder = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        resultFolder = f"./savedModels/{resultFolder}/"
        os.makedirs(resultFolder, exist_ok=True)
        print(f"Saving models to: {resultFolder}")

        log_path = os.path.join(resultFolder, "episode_log.csv")
        with open(log_path, "w") as f:
            f.write("episode,total_reward,loss,epsilon,steps\n")

        

        results = []
        for ep in range(num_episodes):
            frequency = 50
            save_images = (ep % frequency == 0)
            print_status = (ep % frequency/2 == 0)
            result = self.run_episode(
                print_status=print_status,
                save_images=save_images,
            )
            results.append(result)

            if ep % save_models_every == 0:
                self.Q.saveModel(f"{resultFolder}/{ep}.pth")

            print(f"Episode {ep}, return={result['cumulativeReward']}")

        return results
