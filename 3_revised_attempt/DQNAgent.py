# figures/processing
import pandas as pd
import numpy as np

# saving outputs
import os       # making folders
import datetime # specific run
import csv      # results file
import json     # configuration

# environment
import gym_super_mario_bros 
import nes_py      
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

# epsilon, action selection
from random import randint, random

# neural networks
import torch
import torch.optim as optim


import helpers
from DQN import DQN
from replay_buffer import ReplayBuffer

# combinations for a complete actionSpace
from itertools import combinations

# deque for stacking of frames
from collections import deque

class DQNAgent():

    def __init__(self, 
                 device = 'cpu',
                 rom = 'v3',
                 stagesList = ['1-1'],
                 excludeList = None,
                 buttonList = [['right']],
                 episode = 0,
                 bufferCapacity = 5_000,                 
                 BATCH_SIZE = 32,
                 GAMMA = 0.99,
                 lr = 1e-4,
                 skipFrames = 4,
                 saveImageFrequency = 250,
                 randomLevel = True, 

                 note = None, # additional note for json config
                 debug = True,
                ):

        self.config = dict(
            #device=device,
            rom=rom,
            stagesList=stagesList,
            excludeList = excludeList,
            buttonList=buttonList,
            randomLevel=randomLevel,
            bufferCapacity=bufferCapacity,
            BATCH_SIZE=BATCH_SIZE,
            GAMMA=GAMMA,
            lr=lr,
            skipFrames=skipFrames,
            saveImageFrequency=saveImageFrequency,
            startingEpisode=episode,
            note = note
        )



        #self.D = replayBuffer  
        self.device = device
        self.episode = episode
        self.BATCH_SIZE = BATCH_SIZE
        self.minBufferSize = 5 * self.BATCH_SIZE
        self.GAMMA = GAMMA       
        self.randomLevel = randomLevel
        self.rom = rom
        self.skipFrames = skipFrames
        self.saveImageFrequency = saveImageFrequency
        self.buttonList = buttonList # name is descriptive and intentional, this doesn't become an actionSpace until it comes out from gymnasium
        self.stagesList = stagesList
        


        # constants
        self.FRAME_HEIGHT = 240
        self.FRAME_WIDTH = 256
        self.VTRIM = 36
        self.HTRIM = 36       # trim pixels off the left
        self.HTRIM_RIGHT = 16 # trim pixels off the right
        self.TRIM_FRAME_HEIGHT = self.FRAME_HEIGHT - self.VTRIM
        self.TRIM_FRAME_WIDTH  = self.FRAME_WIDTH  - self.HTRIM - self.HTRIM_RIGHT
        self.ADJ_FRAME_HEIGHT = 100 # downscaled from trimmed
        self.ADJ_FRAME_WIDTH = 100  # 

        # environment initialization
        print(f"Initializing gymnasium environment ({rom})")
        self.env, self.actionSpace = helpers.initializeEnvironment(stagesList = self.stagesList, 
                                                                   buttonList = self.buttonList,
                                                                   rom = self.rom)
      
        # set up Q and D (Q network and replay buffer) 
        
        self.observationShape = (3 * self.skipFrames, self.ADJ_FRAME_HEIGHT, self.ADJ_FRAME_WIDTH)
        print("Setting up Replay Buffer")
        self.D = ReplayBuffer(bufferCapacity, self.observationShape) # device=device)
        print("Setting up DQN")        
        self.Q = DQN(self.observationShape, len(self.actionSpace)).to(self.device)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr = lr)
        
        # set up results folder on a per-agent basis, ie., per run
        print("Setting up output folders...")
        self.startTime = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.resultsDir = f'./results/{self.startTime}/'
        self.savedModelsDir = os.path.join(self.resultsDir, 'savedModels')
        self.savedSequencesDir = os.path.join(self.resultsDir, 'savedSequences')
        self.logfile = os.path.join(self.resultsDir, './log.csv')
        os.mkdir(self.resultsDir)
        os.mkdir(self.savedModelsDir)
        os.mkdir(self.savedSequencesDir)
        print(f"Saving outputs to : {self.resultsDir}")
        
        config_path = os.path.join(self.resultsDir, "config.json")
        print(f"Saving configuration to: {config_path}")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

        # per-episode rewards and other information
        self.logPath = os.path.join(self.resultsDir, "log.csv")
        print(f"Saving log to: {self.logPath}")

        with open(self.logPath, "w") as f:
            header = "episode,total_reward,x_pos,time,flag_get,loss,epsilon,steps,course,time\n"
            f.write(header)

    
    def __repr__(self):

        s = ''
        s += f"{self.stagesList = }\n"
        s += f"{self.GAMMA = }\n"
        s += f"{self.BATCH_SIZE = }\n"
        s += f"{self.episode = }\n"

        return(s)
        
    def selectAction(self, phi):
        # select a_t = max_a Q∗(φ(st), a; θ)
        ##phi = phi.unsqueeze(0) # (c, w, h) -> (1, c, w, h)
    
        if random() < 0.01:
            return(randint(0, len(self.buttonList) - 1)) #randint is inclusive of right   
        else:
            # get the argmax output from Q-network
            # cast as integer to pass into gymnasium
            return(torch.argmax(self.Q(phi)).item())
    
    ## generated by generative AI, minor edits
    def compute_targets(self, minibatch):
        phi_batch, action_batch, reward_batch, next_state_batch = minibatch
    
        # print(f"{action_batch.dtype =}")
        # print(f"{action_batch[0] = }")
    
        # Current Q-values for chosen actions
        q_values = self.Q(phi_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
    
        # Next state Q-values
        next_q_values = self.Q(next_state_batch)
        max_next_q = next_q_values.max(dim=1)[0]
    
        # Targets
        y = reward_batch + self.GAMMA * max_next_q
    
        return q_values, y
    ## end generative AI

    def runEpisode(self,
                   seed = None,
                   saveImage = False,
                   debug = False,
                   epoch = None,
                  ):
    
        """ Run a single episode until death. """
    
        cumulativeReward = 0
        maxEpisodeLength = 20_000 # in testing, this seems to be reasonable, even accounts for degenerate puzzle course cases
        actions = np.empty(maxEpisodeLength, dtype = int)
        actions.fill(-1)
        
        # set up separate folders for raw and preprocessed images
        if saveImage:
        
        state, info = self.env.reset(seed = seed) if seed else self.env.reset()
        if state is None:
            print("state is none!?")
            rawDir = f'{self.savedSequencesDir}/raw/{self.episode}'
            preproDir = f'{self.savedSequencesDir}/preprocessed/{self.episode}'
            os.makedirs(rawDir, exist_ok = True)
            os.makedirs(preproDir, exist_ok = True)
        phi = helpers.preprocessFrame(state)
        
        step = -1
        
        while True:
            step += 1
    
            # epsilon-greedy or otherwise select a_t = max_a Q^*(φ(st), a; θ) 
            action = self.selectAction(phiT)
            actions[step] = action

            # TODO: remove
            try:
                action_text = self.actionSpace[action]
            except IndexError:
                print(action, self.actionSpace)
    
            state, reward, terminated, truncated, info = self.env.step(action)
            phiPrime = helpers.preprocessFrame(state)
            phiPrimeT = helpers.tensorify(phiPrime)
    
            # store new transition
            self.D.storeTransition( (phiT, action, reward, phiPrimeT) )
            cumulativeReward += reward

            phiT = phiPrimeT
    
            train_frequency = 4 # train every 'x' frames
            if step % train_frequency == 0:
                # sample random minibatch of transitions (φ_j, a_j, r_j, φ_{j+1}) from D          
                minibatch = self.D.sample(self.BATCH_SIZE)
                # y = r_j + γ max_{a′} Q(φ_{j+1}, a′; θ) for non-terminal φ_{j+1} 
                # for the moment ignoring the terminal state
                q_values, y = self.compute_targets(minibatch)
                loss = torch.nn.functional.mse_loss(q_values, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # imgState = state.squeeze(0)
                # imgPhi = phi.squeeze(0)
                
    
        result = {}
        result['step'] = step
        result['loss'] = 0 # loss
        result['cumulativeReward'] = cumulativeReward
        result['info'] = info
        
        return(result)
    
    
    def runEpisodes(self, numEpisodes):
    
            
            # Extract fields
            total_reward = result["cumulativeReward"]
            loss = result["loss"]
            course = f"{result['info']['world']}-{result['info']['stage']}"
            flag_get = f"{result['info']['flag_get']}"
            x_pos = f"{result['info']['x_pos']}"
            game_time = f"{result['info']['time']}"
            steps = result['step']
            time = datetime.datetime.now()

            # write results to CSV
            with open(self.logPath, "a") as f:
                f.write(f"{self.episode},{total_reward},{x_pos},{game_time},{flag_get},{loss},{self.epsilon},{steps},{course},{time}\n")        

            if i % 1_000 == 0 or i == 0:
                modelfp = os.path.join(self.savedModelsDir, f"{i}.pth")
                self.Q.saveModel(modelfp)
        return()