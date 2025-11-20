# see Mnih et al. 2013 Playing Atari with Deep Reinforcement Learning
# Algorithm 1 for reference

import torch

class ReplayBuffer():
    def __init__(self, size, shape):
        """ Initialize a replay buffer of a fixed size
        Note, that this is not going to prefill the buffer with experiences.
        That will be left to external logic to prefill."""

        # initial thinking was to use a deque, which would be convenient
        # but, it would likely be more coherent to use a collection of tensors
        # this approach was suggested through a Socratic prompt to an LLM

        self.size  = size
        self.shape = shape
        
        self.phi       = torch.zeros( (size, *shape) )
        self.action    = torch.zeros( (size) )
        self.reward    = torch.zeros( (size) )
        self.nextState = torch.zeros( (size, *shape) )

        self.probabilities = torch

        self.index = 0 # pointer to the **next** entry to be filled in

    def storeTransition(self, transition):
        """ Store a transition of state, action, reward, state' aka (phi, a, r, phi+1) """

        self.phi[self.index]       = transition[0]
        self.action[self.index]    = transition[1]
        self.reward[self.index]    = transition[2]
        self.nextState[self.index] = transition[3]

        # will loop through in a circular manner, First-In-First-Overwritten
        self.index = (self.index + 1) % self.size

        
    def sample(self, minibatches = 1):
        """ Sample a minibatch of transitions. 
        Input: 
            number of samples
        Output:
            tuple of (phi, a, r, phi+1)
                preprocessed frame
                action
                reward
                next preprocessed frame """

        # we could certainly use numpy or similar
        # unclear if there is a torch specific option
        # looking at torchrl source code for samplers suggests
        # torch.randperm
        ## see: https://github.com/pytorch/rl/blob/main/torchrl/data/replay_buffers/samplers.py
        #

        # original source does not specify with or without replacement
        # as is, this does not allow replacement
        indices = torch.randperm(self.size)[0 : minibatches]

        result = (
            self.phi[indices],
            self.action[indices],
            self.reward[indices],
            self.nextState[indices]
        )

        return(result)
        

    def __repr__(self):
        s = ''
        s += f"{self.size = }\n"
        s += f"{self.shape = }\n"
        s += f"{self.index = }\n"

        return(s)