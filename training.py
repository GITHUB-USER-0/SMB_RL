# training/main loop skeleton?

"""
Initialize replay memory D to capacity N
Initialize action-value function Q with random weights 

for episode = 1, M do 
    Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1)
    for t = 1, T do  
        With probability ϵ select a random action a_t
        otherwise select a_t = max_a Q^*(φ(st), a; θ) 
        Execute action a_t in emulator and observe reward r_t and image x_{t+1}
        Set s_{t+1} = s_t, a_t, x_{t+1} and preprocess φ_{t+1} = φ(s_{t+1})
        Store transition (φ_t, a_t, rt, φt+1) in D 
        Sample random minibatch of transitions (φ_j, a_j, r_j, φ_{j+1}) from D  
        Set y_j =  { r_j for terminal φ_{j+1}
                     r_j + γ max_{a′} Q(φ_{j+1}, a′; θ) for non-terminal φ_{j+1} 
        Perform a gradient descent step on (yj − Q(φj, aj; θ))2 according to equation 3
    end for 
end for

"""
# ------------------------------------
# libraries

# figures/processing
import pandas as pd
import numpy as np

# logging
import csv

# environment
import gym_super_mario_bros 
import nes_py      
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

# saving outputs
# from time import monotonic  
import os
#from pathlib import Path # https://stackoverflow.com/questions/273192/how-do-i-create-a-directory-and-any-missing-parent-directories
import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageOps import grayscale
from PIL.ImageOps import fit as resize

from random import randint, random

import sys #argument handling

#from collections import deque
import torch
import torch.optim as optim
import torchvision # grayscale

import DQN
from replay_buffer import ReplayBuffer
from DQN import DQN
from itertools import combinations # combinations for a complete actionSpace

# ------------------------------------
# functions
def prefillBuffer(BUFFER_SIZE, env, actionSpace):
    """ prefill the replay buffer

    Note that if random levels are selected outside this function, it is likely that the 
    buffer size may need to be much larger than if one focused on just one level.
    """

    # grayscale
    rb = ReplayBuffer(BUFFER_SIZE, (1, ADJ_FRAME_HEIGHT, ADJ_FRAME_WIDTH) )

    # need an environment to get transitions
    state, info = env.reset(seed = seed) if SEED else env.reset()

    # uniform random action selectio...n
    # it may be worth pre-weighting here with TAS inputs
    # see 'on TAS and ROMs.ipynb'
    # Space.sample(probability) as per: https://gymnasium.farama.org/api/spaces/
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())


    resetCount              = 0
    transitionCount         = 0 # counter of transitions filled into buffer
    transitionsCurrentLevel = 0 # counter of per-level transitions, 
                                # helps to avoid having too many transitions from one level
    while transitionCount < BUFFER_SIZE:
        
        old_state = state
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())

        # reset environment if dead, or too many samples from current environment
        if terminated or truncated or transitionsCurrentLevel % (BUFFER_SIZE // 10) == 0:
            state, info = env.reset(seed = seed) if SEED else env.reset()
            state, reward, terminated, truncated, info = env.step(env.action_space.sample())
            resetCount += 1
            transitionsCurrentLevel = 0
            next
            
        rb.storeTransition((
            preprocessFrame(old_state),
            torch.randint(low = 0, high = env.action_space.n, size = (1,)),
            torch.randint(high = 10, size = (1,)),   
            preprocessFrame(state)
        ))

        transitionCount += 1
        transitionsCurrentLevel += 1

        if BUFFER_SIZE >= 100:
            if i % (BUFFER_SIZE // 10) == 0:
                print(f"Filling buffer slot {i} of {BUFFER_SIZE}")
    print(f"Filling buffer sampled from {resetCount} level starts (not guaranteed unique)") 

    return(rb)




class DQNAgent():

    def __init__(self, 
                 randomLevel = True, 
                 rom = 'v3',
                 buttonList = [['NOOP']],
                 debug = True
                ):

        # environment initialization
        self.randomLevel = randomLevel
        self.rom = 'v0'
        self.buttonList = buttonList

        # curated list of levels
        self.stagesList = ['6-3']

        # constants
        self.FRAME_HEIGHT = 240
        self.FRAME_WIDTH = 256
        self.VTRIM = 36
        self.HTRIM = 36 # trim pixels off the left
        self.HTRIM_RIGHT = 16 # trim pixels off the right
        self.TRIM_FRAME_HEIGHT = self.FRAME_HEIGHT - self.VTRIM
        self.TRIM_FRAME_WIDTH  = self.FRAME_WIDTH  - self.HTRIM - self.HTRIM_RIGHT
        self.ADJ_FRAME_HEIGHT = 100 # downscaled from trimmed
        self.ADJ_FRAME_WIDTH = 100  # 
    
    def initializeEnvironment(self):
        """Initialize environment in gymnasium. 
        Sets the list of acceptable actions.
    
        Inputs:
            mode - the set of available actions, either from JoypadSpace, or
                   a custom set of actions provided as a list of lists of actions
            rom - the selected ROM
        Outputs:
            (env, actionSpace_init) - a tuple of the gymnasium environment and actionSpace
    
        Note that v0 offers a traditional view, corresponding 
        to 'super-mario-bros.nes' included with package
        with MD5 of: 673913a23cd612daf5ad32d4085e0760
        and is "Super Mario Bros. (E) (REVA) [!].nes SourceDB: GoodNES 3.23"
        as per: https://tasvideos.org/Games/1/Versions/List
        v3, in turn, is a simplified rectangular view
        this may have been generated by kautenja
        and does not appear in the TAS collections that I saw
    
        v0 is more visually appealing, but it seems plausible that v3
        would train faster.
        """
        if self.rom == 'v0':
            s = 'SuperMarioBros-v0'
        elif self.rom == 'v3':
            s = 'SuperMarioBros-v3'
        else:
            print("Error in ROM selection.")
            return(None)
    
        # as per documentation, SuperMarioBrosRandomStages-v0 will randomly select world, level combinations
        if self.randomLevel:
            s = s.split('-')
            s = 'RandomStages-'.join(s)

        if self.stagesList:
            env = gym_super_mario_bros.make(s, stages = self.stagesList)
        else:
            env = gym_super_mario_bros.make(s)
    
        if self.buttonList == "simple":
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            actionSpace_init = SIMPLE_MOVEMENT
        elif self.buttonList == "complex":
            env = JoypadSpace(env, COMPLEX_MOVEMENT)
            actionSpace_init = COMPLEX_MOVEMENT
        elif self.buttonList == "rightOnly":
            env = JoypadSpace(env, RIGHT_ONLY)
            actionSpace_init = RIGHT_ONLY
        else:
            # provide a predefined list of string actions
            # eg., [['NOOP'], ['right', 'A'], ['right', 'B'], ['right', 'B', 'A']]
            env = JoypadSpace(env, self.buttonList)
            actionSpace_init = self.buttonList
    
        return( (env, actionSpace_init) )

    # arguments
    def preprocessFrame(self, frame):
        """ Preprocess an input image frame
    
        Inputs:
            frame - frame of RGB image data, from the environment
                    a numpy array of shape (240, 256, 3)
                                      Height  x Width  x Channels
                                          Rows  x Col x Channels
    
        Note that the stacking of sequential frames, as per Mnih et al. 2013 
        is to be handled separately.
        """
        
        # might be interesting to benchmark this preprocessing--how much time is spent on this part of the pipeline
    
        frame = frame[self.VTRIM:, :, :]
        frame = frame[:, self.HTRIM:self.FRAME_WIDTH - self.HTRIM_RIGHT, :]
    
        # convert to PIL image
        frame = Image.fromarray(frame)

        # grayscale
        # there is an argument to *not* use grayscale
        # Consider courses that use a gray palette in which
        # mario is near indistinguishable from background
        frame = grayscale(frame)
    
        # downscale
        frame = resize(frame, (ADJ_FRAME_HEIGHT, ADJ_FRAME_WIDTH) )
    
        # convert back to numpy array
        frame = np.array(frame)

        # normalize -- necessary?
        frame = frame / 255.0
        
        return(frame)

    




def emitConfig():
    # from global
    vars = [FRAME_WIDTH, VTRIM, HTRIM, HTRIM_RIGHT, TRIM_FRAME_HEIGHT, TRIM_FRAME_WIDTH, ADJ_FRAME_HEIGHT, ADJ_FRAME_WIDTH, BUFFER_SIZE, SEED, ROM, BATCH_SIZE, GAMMA, LEARNING_RATE, BASIC_ACTION_SPACE, ACTION_SPACE_IN_USE]
    var_labels = ["FRAME_WIDTH", "VTRIM", "HTRIM", "HTRIM_RIGHT", "TRIM_FRAME_HEIGHT", "TRIM_FRAME_WIDTH", "ADJ_FRAME_HEIGHT", "ADJ_FRAME_WIDTH", "BUFFER_SIZE", "SEED", "ROM", "BATCH_SIZE", "GAMMA", "LEARNING_RATE", "BASIC_ACTION_SPACE", "ACTION_SPACE_IN_USE"]

    with open('./results/log.txt', 'a') as o:
        o.write("\n---\nStarting new set of episodes\n")
        o.write(f"{datetime.datetime.now()}\n")
        
        for var, label in zip(vars, var_labels):
            # separate label necessary {var = } will print 'var'
            o.write(f"{label} = {var}\n") 
            
# ------------------------------------
# constants
FRAME_HEIGHT = 240
FRAME_WIDTH = 256
VTRIM = 36
HTRIM = 36 # trim pixels off the left
HTRIM_RIGHT = 16 # trim pixels off the right
TRIM_FRAME_HEIGHT = FRAME_HEIGHT - VTRIM
TRIM_FRAME_WIDTH  = FRAME_WIDTH  - HTRIM - HTRIM_RIGHT
ADJ_FRAME_HEIGHT = 100 # downscaled from trimmed
ADJ_FRAME_WIDTH = 100  # 
BUFFER_SIZE = 1_000
SEED = None
ROM = 'v0'
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4
BASIC_ACTION_SPACE = [['right'], ['left'], ['down'], ['up'], ['B'], ['A'], ['NOOP']]

ACTION_SPACE_IN_USE = [['right'], ['NOOP'], ['right', 'B'], ['right', 'A'], ['down']]
    

# ------------------------------------
# action spaces
# see action_space_and_TAS.ipynb
ALL_BUTTONS = ['right', 'left', 'down', 'up', 'start', 'select', 'B', 'A']
ALL_SINGLE_ACTIONS = ALL_BUTTONS.copy()
ALL_SINGLE_ACTIONS.append("NOOP")
COMPLETE_ACTION_SPACE = [ ["NOOP"] ] # note use of a list of lists 

for i in range(1, len(ALL_BUTTONS) + 1): # do not include the empty permutation
                                         # already included above
    for j in combinations(ALL_BUTTONS, i):
        COMPLETE_ACTION_SPACE.append(list(list(j)))


# ------------------------------------
# training

if __name__ == "__main__":

    import sys
    sys.exit()

    if len(sys.argv) > 1:
        print("Received additional options:", sys.argv[1:])

        try:
            if sys.argv[1][-4:] == '.pth':
                filename = sys.argv[1]
                print(f"Interpreting {filename} as a model to load instead of a randomly initialized DQN.")
        except IndexError:
            filename = None
            pass
    
    print(f"Original frame size (H x W): {FRAME_HEIGHT} x {FRAME_WIDTH}")
    print(f"Trimmed frame size (H x W): {TRIM_FRAME_HEIGHT} x {TRIM_FRAME_WIDTH}") 
    print(f"Downscaled frame size (H x W): {ADJ_FRAME_HEIGHT} x {ADJ_FRAME_WIDTH}")
    
    
    ## Initialize action-value function Q with random weights 
    print("\n---\nInitializing Deep Q Network")
    # Q = DQN(actionSpaceSize = len(BASIC_ACTION_SPACE),
    #         RGB = False,
    #         width  = ADJ_FRAME_WIDTH,
    #         height = ADJ_FRAME_HEIGHT)

    Q = DQN( input_shape = (1, ADJ_FRAME_WIDTH, ADJ_FRAME_HEIGHT),
             num_actions = len(ACTION_SPACE_IN_USE))
    
    print("Q-network info")
    print(f"{Q =}")
    print("A sample output's shape from the feature extractor: ", Q.featureExtractor(torch.zeros(1,1,ADJ_FRAME_WIDTH,ADJ_FRAME_HEIGHT)).shape)
    
    optimizer = optim.Adam(Q.parameters(), lr=LEARNING_RATE)
    
    print("\n---\nInitializing gymnasium environment")
    env, actionSpace = initializeEnvironment(ACTION_SPACE_IN_USE, rom = ROM)
    print(f"environment initialized with rom: {ROM}")
    
    ## Initialize replay memory D to capacity N
    print("\n---\nInitializing replay memory buffer")
    D = prefillBuffer(BUFFER_SIZE, env, actionSpace)
    print(f"Buffer filled with {BUFFER_SIZE} transitions.")