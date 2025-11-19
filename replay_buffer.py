# see Mnih et al. 2013 Playing Atari with Deep Reinforcement Learning
# Algorithm 1 for reference

class ReplayBuffer():
    def __init__(self, size):
        """ Initialize a replay buffer of a fixed size """
        pass

    def storeTransition():
        """ Store a transition of state, action, reward, state' aka (phi, a, r, phi+1) """
        
        pass

    def sample():
        """ Sample a minibatch of transitions. 
        Input: 
            number of samples
        Output:
            tuple of (phi, a, r, phi+1)
                preprocessed frame
                action
                reward
                next preprocessed frame """
        pass