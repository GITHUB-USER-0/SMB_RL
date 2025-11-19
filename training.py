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