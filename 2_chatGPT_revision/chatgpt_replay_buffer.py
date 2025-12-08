# replay_buffer.py
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device='cpu'):
        """
        capacity: max number of transitions
        obs_shape: (C, H, W)
        """
        self.capacity = capacity
        self.device = device

        self.states      = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
        self.actions     = torch.zeros((capacity,), dtype=torch.int64)
        self.rewards     = torch.zeros((capacity,), dtype=torch.float32)
        self.next_states = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
        self.dones       = torch.zeros((capacity,), dtype=torch.float32)

        self.idx = 0
        self.size = 0

    def store(self, state, action, reward, next_state, done):
        """Store one transition (s, a, r, s', done)."""
        self.states[self.idx].copy_(state.squeeze(0))      # (C,H,W)
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx].copy_(next_state.squeeze(0))
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Return a batch of transitions as tensors on self.device."""
        assert self.size > 0, "Replay buffer is empty."
        indices = torch.randint(0, self.size, (batch_size,))

        s  = self.states[indices].to(self.device)
        a  = self.actions[indices].to(self.device)
        r  = self.rewards[indices].to(self.device)
        sp = self.next_states[indices].to(self.device)
        d  = self.dones[indices].to(self.device)

        return s, a, r, sp, d

    def __len__(self):
        return self.size
