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
def save_diagnostic_image(frameArray, step, action, x_pos, y_pos, img_dir):
    image = Image.fromarray(frameArray.copy())
    draw = ImageDraw.Draw(image)

    text_annotation = ""
    text_annotation += f"step: {step:0>6}\naction: {action}\n"
    text_annotation += f"x: {x_pos:0>3}, y: {y_pos:0>3}\n"

    draw.text((0, 0), text_annotation, fill=(0,0,0))

    image.save(os.path.join(img_dir, f"{step:0>6}_{monotonic()}.png"))
    

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


def preprocess_frame(frame):
    """Convert RGB image to grayscale and downsample."""
    img = Image.fromarray(frame)
    img = img.convert('L')       # grayscale
    img = img.resize((84, 84))   # reduce size
    return np.array(img).astype(np.float32) / 255.0

class ReplayBuffer:
    def __init__(self, size=50000):
        self.buffer = deque(maxlen=size)

    def push(self, s, a, r, s2, done):
        # Convert numpy frames → torch tensors (1×84×84)
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).unsqueeze(0).float()  # shape: [1,84,84]
        if isinstance(s2, np.ndarray):
            s2 = torch.from_numpy(s2).unsqueeze(0).float()

        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.stack(s),
                torch.tensor(a, dtype=torch.int64),
                torch.tensor(r, dtype=torch.float32),
                torch.stack(s2),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)


def create_level_folders(world, stage):
    base_dir = f"./runs/World-{world}-{stage}"
    img_dir = os.path.join(base_dir, "images")

    os.makedirs(img_dir, exist_ok=True)
    return base_dir, img_dir

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        C, H, W = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # ***** FIX: dynamically compute FC input size *****
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.conv(dummy)
            conv_out_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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

        self.base_dir, self.img_dir = create_level_folders(world, stage)
        self.reward_log_path = os.path.join(self.base_dir, "rewards.csv")

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actionSpace = actionSpace

        # ---- Fix: determine input shape dynamically ----
        obs, _ = self.env.reset()
        state = self.preprocess(obs)
        input_shape = state.shape  # (C, H, W)

        n_actions = len(self.actionSpace)

        # ---- Correct DQN init ----
        self.policy_net = DQN(input_shape, n_actions).to(self.device)
        self.target_net = DQN(input_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(size=10000)

    # -------------------------------------------------
    # Preprocess raw frame into tensor usable by DQN
    # -------------------------------------------------
    def preprocess(self, obs):
        from PIL import Image
        obs = obs.copy()  # fix negative stride issue
        img = Image.fromarray(obs).convert("L")  # grayscale
        img = img.resize((84, 84))
        obs = torch.tensor(np.array(img), dtype=torch.float32, device=self.device) / 255.0
        obs = obs.unsqueeze(0)  # add channel dimension: (1,H,W)
        return obs



    # -------------------------------------------------
    # ε-greedy action selection
    # -------------------------------------------------
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(len(self.actionSpace))

        with torch.no_grad():
            state = state.unsqueeze(0)  # add batch dimension
            q_values = self.policy_net(state)

        return int(torch.argmax(q_values))

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # -------------------------------------------------
    # Replay training step
    # -------------------------------------------------
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        q_vals = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values
        with torch.no_grad():
            next_q_vals = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q_vals * (1 - dones)

        loss = nn.MSELoss()(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    def train(self, episodes=2000, batch_size=32, max_steps=2000):
        rewards = []

        with open(self.reward_log_path, "w") as f:
            f.write("episode,reward\n")

        for ep in range(episodes):
            #torch.cuda.empty_cache()
            obs, _ = self.env.reset()
            state = self.preprocess(obs)

            total_reward = 0

            for t in range(max_steps):
                action = self.act(state)

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocess(next_obs)

                # Store in replay buffer
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Save diagnostic images periodically
                if t % 10 == 0:
                    save_diagnostic_image(
                        next_obs,
                        step=t,
                        action=action,
                        x_pos=info.get("x_pos", 0),
                        y_pos=info.get("y_pos", 0),
                        img_dir=self.img_dir
                    )

                # Train
                self.replay(batch_size)

                # Update target
                if t % 500 == 0:
                    self.update_target()

                if done:
                    break

            rewards.append(total_reward)
            if ep % 100 == 0:
                print(f"Episode {ep+1}/{episodes} | Reward={total_reward:.1f} | Epsilon={self.epsilon:.3f}")
            with open(self.reward_log_path, "a") as f:
                f.write(f"{ep+1},{total_reward}\n")

        return rewards
