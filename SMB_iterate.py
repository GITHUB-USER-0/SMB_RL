import gym_super_mario_bros
import nes_py
import pandas as pd
import numpy as np
from time import monotonic
import os
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import copy as copycopy # for deep copying? see: 
# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html#numpy.ndarray.copy

from itertools import combinations # combinations for a complete actionSpace


from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from random import seed as randomSeed
from random import shuffle


import torch
import torch.nn as nn
import torch.optim as optim
import random


# see action_space_and_TAS.ipynb
ALL_BUTTONS = ['right', 'left', 'down', 'up', 'start', 'select', 'B', 'A']
ALL_SINGLE_ACTIONS = ALL_BUTTONS.copy()
ALL_SINGLE_ACTIONS.append("NOOP")
COMPLETE_ACTIONSPACE = [ ["NOOP"] ] # note use of a list of lists 
for i in range(1, len(ALL_BUTTONS) + 1): # do not include the empty permutation
                                         # already included above
    for j in combinations(ALL_BUTTONS, i):
        COMPLETE_ACTIONSPACE.append(list(list(j)))


# does not seem happy to work in a tensorflow kernel
def save_diagnostic_image(frameArray, step, action, x_pos, y_pos):
    """ Save a snapshot with additional text info burned in."""
    image = Image.fromarray(frameArray.copy())
    
    draw = ImageDraw.Draw(image)
    ## fails in TensorFlow environment
    #font = ImageFont.truetype("arial.ttf", size = 20)
    text_annotation = ""
    text_annotation += str(f"step: {step:0>6}\naction: {action}\n")
    text_annotation += str(f"x: {x_pos:0>3}, y: {y_pos:0>3}\n")

    #white_text = (255, 255, 255)
    black_text = (0,   0,   0  )
    draw.text((0, 0), text_annotation, fill = black_text)

    # use of padding in filename is helpful for passing 
    # in to Kdenlive as an Image Sequence for video review
    # in quick testing, .png was actually smaller than .jpeg
    image.save(f"./states/{step:0>6}_{monotonic()}.png")
    

def initialize_environment(world=1, stage=1, mode="simple", rom_version="v3"):
    print(world, stage, mode, rom_version)
    env_id = f"SuperMarioBros-{world}-{stage}-{rom_version}"
    env = gym_super_mario_bros.make(env_id)

    if mode == "simple":
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        actionSpace = SIMPLE_MOVEMENT
    elif mode == "complex":
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        actionSpace = COMPLEX_MOVEMENT
    elif mode == "rightOnly":
        env = JoypadSpace(env, RIGHT_ONLY)
        actionSpace = RIGHT_ONLY
    else:
        env = JoypadSpace(env, mode)
        actionSpace = mode

    return env, actionSpace


class Agent():
    
    def __init__(self, mode="simple", world=1, stage=1, rom_version="v3", seed=5004):

        self.env, self.actionSpace = initialize_environment(
            world=world,
            stage=stage,
            mode=mode,
            rom_version=rom_version
        )

        # tracking helpful measures for diagnostics/debugging
        self.x_positions = []
        self.y_positions = []
        self.actions = []
        self.timePoints = [(monotonic(), 0)] # collect the start point, and append any new lives

        self.step = -1
        self.cumulativeReward = 0
        self.terminated = False
        self.truncated = False
        self.prior_time = None
        self.state = None
        self.seed = seed

        # a number of states to collect for manual review
        queueLength = 500
        self.trailingStates = deque() # double-ended queue
        
        # pre-populate to avoid edge cases, like popping from empty
        for i in range(0, queueLength):
            self.trailingStates.append(None)

        # check for whether seed is specified?
        # this unfortunately is presently inadequate to guarantee stability
        state = self.env.reset(seed = self.seed)
        np.random.seed(self.seed)
        randomSeed(self.seed)        

    def __repr__(self):
        result = ""
        length = len(self.actionSpace)
        if length <= 18: #18 is the length of the Happy Lee TAS actionSpace
            result += f"{self.actionSpace=}\n"
        else:
            result += f"actionSpace is long, (length: {length}), showing first 5 and last 5 entries:\n"
            result += f"{str(self.actionSpace[0:5])}\n ... \n {str(self.actionSpace[-5:])}\n"
        result += f"{self.seed=}\n"
        result += f"{self.step=}\n"
        result += f"{self.cumulativeReward=}\n"
        result += f"Latest state:\n{self.state}"

        return(result)
    
    def iterate(self, policy, maxSteps = None, saveImage = False):
        if isinstance(policy, list):
            sequence = True
            policyIndex = 0

            # don't go beyond policy
            # may be interesting to switch from rigid sequence to policy
            # partway through an episode, but not currently in scope
            maxSteps = len(policy)

        
        while True:
            self.step += 1

            # break early
            if maxSteps is not None:
                if self.step >= maxSteps:
                    break

            if policy == "random":
                # env.action_space is discrete, a range over the length of the number of actions
                # may be simpler to specify otherwise to avoid confusion with actionSpace
                action = self.env.action_space.sample() # as integer index
                action_text = self.actionSpace[action] # for diagnostic
            elif sequence:
                # could do with renaming, but at this point the policy *is* a list of inputs
                action_text = policy[policyIndex]
                action = self.actionSpace.index(action_text)
                #action = self.env.action_space.index(action_text)
                policyIndex += 1
            
            self.state, reward, terminated, truncated, info = self.env.step(action)
        
            if terminated or truncated:
                print(f"{terminated=}\n{truncated=}")
                print(f"{info=}")
                break
        
            # diagnostic info collection
            self.cumulativeReward += reward
            self.x_positions.append(info['x_pos'])
            self.y_positions.append(info['y_pos'])
            self.actions.append(action_text)
            # remove first, add to end of deque
            self.trailingStates.popleft()
            self.trailingStates.append(self.state.copy()) # copy it over!
        
            # diagnostic info printing
            # print periodically, and as mario is timing out
            if self.step % 1_000 == 0 or info['time'] <= 0 :
                print(f"{self.step=:0>7}, {self.cumulativeReward=}, {info['coins']=}, {info['time']=}")
            
            # collect information regarding end of life, time points
            if self.prior_time is not None:
                if self.prior_time < info['time']:
                    print("start of new life")
                    print("current lives: ", info['life'])
                    self.timePoints.append( (monotonic(), self.step) )
                    break # just do one life
                self.prior_time = info['time']
            if self.prior_time is None:
                self.prior_time = info['time']

            if saveImage:
                save_diagnostic_image(self.state, self.step, action_text, info['x_pos'], info['y_pos'])
        
        self.env.close()

    def actionDF(self):
        """A simple dataframe overview of actions taken."""
        
        df = pd.DataFrame(self.actions)

        # variable numbers of possible simultaneous actions
        for index, col in enumerate(df.columns):
            df = df.rename({
                index : 'a' + str(index)
            }, axis = 1)
        
        df = df.fillna('') # None to ''
        df['action'] = ''
        
        for col in df.columns:
            if col[0] == 'a' and col != 'action':
                df['action'] = df['action'] + ',' + df[col]
                
        df['action'] = df['action'].str.strip(',')
        df['buttonCount'] = df['action'].str.count(',') + 1
        df['buttonCount'] = df.apply(lambda x : 0 if x['action'] == "NOOP" else x['buttonCount'], axis = 1)
        df = df.astype({'buttonCount' : 'int32'})
        
        return(df)

class QLearningAgent(Agent):
    def __init__(self, actionSpace, world=1, stage=1, rom_version="v3", seed=5004, alpha=0.1, gamma=0.99, epsilon=0.1):

        self.env, self.actionSpace = initialize_environment(
            world=world,
            stage=stage,
            mode=actionSpace,
            rom_version=rom_version
        )
        
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.Q = {}             # state-action table

    def get_state(self, info):
        """Discretize the state into a manageable tuple."""
        x = info['x_pos'] // 40        # bin every ~40 pixels
        y = info['y_pos'] // 40
        time_bin = info['time'] // 50  # reduce time precision
        return (x, y, time_bin)

    def get_Q(self, state, action):
        """Return Q-value, default to 0."""
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.actionSpace))  # explore
        else:
            q_values = [self.get_Q(state, a) for a in range(len(self.actionSpace))]
            return np.argmax(q_values)  # exploit

    def update(self, state, action, reward, next_state, done):
        max_next_Q = max([self.get_Q(next_state, a) for a in range(len(self.actionSpace))], default=0)
        target = reward + (0 if done else self.gamma * max_next_Q)
        old_value = self.get_Q(state, action)
        new_value = old_value + self.alpha * (target - old_value)
        self.Q[(state, action)] = new_value

    def train(self, episodes=100):
        rewards = []
        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            info = {'x_pos': 0, 'y_pos': 0, 'time': 400}
            total_reward = 0

            while not done:
                state = self.get_state(info)
                action = self.choose_action(state)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = self.get_state(info)

                self.update(state, action, reward, next_state, done)
                total_reward += reward

            rewards.append(total_reward)
            print(f"Episode {ep+1}/{episodes}, Total Reward: {total_reward}")

        return rewards

def preprocess_frame_old(frame):
    """Convert RGB image to grayscale and downsample."""
    img = Image.fromarray(frame)
    img = img.convert('L')       # grayscale
    img = img.resize((84, 84))   # reduce size
    return np.array(img).astype(np.float32) / 255.0

class DQNAgent(Agent):
    def __init__(self, actionSpace, world=1, stage=1, rom_version="v3", seed=5004, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, lr=1e-3):
        
        self.env, self.actionSpace = initialize_environment(
            world=world,
            stage=stage,
            mode=actionSpace,
            rom_version=rom_version
        )
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.memory = deque(maxlen=5000)

        n_inputs = 84 * 84     # flattened grayscale frame
        n_hidden = 128
        n_actions = len(self.actionSpace)

        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((n_inputs, n_hidden)) * 0.01
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.standard_normal((n_hidden, n_actions)) * 0.01
        self.b2 = np.zeros((1, n_actions))

    def forward(self, state):
        """Forward pass of the Q-network."""
        z1 = state @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        q_values = h1 @ self.W2 + self.b2
        return q_values, h1

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.actionSpace))
        q_values, _ = self.forward(state)
        return np.argmax(q_values)

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        for i in batch:
            s, a, r, s2, done = self.memory[i]
            q_values, h1 = self.forward(s)
            target = q_values.copy()
            q_next, _ = self.forward(s2)
            target[0, a] = r + (0 if done else self.gamma * np.max(q_next))

            # Backprop (simple gradient descent)
            dloss = target - q_values
            dW2 = h1.T @ dloss
            db2 = dloss
            dh1 = dloss @ self.W2.T
            dz1 = dh1 * (h1 > 0)
            dW1 = s.T @ dz1
            db1 = dz1

            # Update weights
            self.W1 += self.lr * dW1
            self.b1 += self.lr * db1
            self.W2 += self.lr * dW2
            self.b2 += self.lr * db2

        # decay exploration
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train(self, episodes=50, max_steps=500, batch_size=32):
        rewards = []
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = preprocess_frame(obs).reshape(1, -1)
            total_reward = 0
            done = False

            for t in range(max_steps):
                action = self.act(state)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = preprocess_frame(next_obs).reshape(1, -1)
                self.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if done:
                    break

                self.replay(batch_size)

            rewards.append(total_reward)
            print(f"Episode {ep+1}/{episodes} - Reward: {total_reward:.1f} - Epsilon: {self.epsilon:.3f}")

        return rewards




def preprocess_frame(frame):
    img = Image.fromarray(frame)
    img = img.convert('L')       # grayscale
    img = img.resize((84, 84))
    arr = np.array(img) / 255.0
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)   # shape: 1×84×84

class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),   # output: 32×20×20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # output: 64×9×9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # output: 64×7×7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, size=50000):
        self.buffer = deque(maxlen=size)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.stack(s),
                torch.tensor(a),
                torch.tensor(r, dtype=torch.float32),
                torch.stack(s2),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

class TorchDQNAgent(Agent):
    def __init__(
        self, 
        actionSpace, 
        world=1, 
        stage=1, 
        rom_version="v3",
        seed=5004,
        gamma=0.99,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.999
    ):
        super().__init__(mode=actionSpace, world=world, stage=stage, rom_version=rom_version, seed=seed)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_actions = len(self.actionSpace)
        self.policy_net = DQN(n_actions).to(self.device)
        self.target_net = DQN(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(size=50000)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(len(self.actionSpace))
        with torch.no_grad():
            q_values = self.policy_net(state.unsqueeze(0).to(self.device))
        return int(torch.argmax(q_values))

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        q_vals = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_vals = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q_vals * (1 - dones)

        loss = nn.MSELoss()(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def train(self, episodes=50, batch_size=32, max_steps=2000):
        rewards = []

        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = preprocess_frame(obs)  # shape 1×84×84
            total_reward = 0

            for t in range(max_steps):

                action = self.act(state)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                next_state = preprocess_frame(next_obs)

                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.replay(batch_size)

                if t % 500 == 0:
                    self.update_target()

                if done:
                    break

            rewards.append(total_reward)
            print(f"Episode {ep+1}/{episodes} | Reward={total_reward:.1f} | Epsilon={self.epsilon:.3f}")

        return rewards
