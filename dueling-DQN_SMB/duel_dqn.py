import pickle
import random
import time
from collections import deque
import csv
import os

import pandas as pd  # <--- NEW: for resuming CSV

import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import *


# -----------------------------
# Helper Functions
# -----------------------------

def arrange(s):
    if not type(s) == np.ndarray:  # <-- fix: was "not type(s) == 'numpy.ndarray'"
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)


class replay_memory(object):
    def __init__(self, N):
        self.memory = deque(maxlen=N)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)


class model(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(model, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(20736, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)

        self.device = device
        self.seq = nn.Sequential(self.layer1, self.layer2, self.fc, self.q, self.v)

        self.seq.apply(init_weights)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - 1 / adv.shape[-1] * adv.sum(-1, keepdim=True))
        return q


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train(q, q_target, memory, batch_size, gamma, optimizer, device):
    s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))
    s = np.array(s).squeeze()
    s_prime = np.array(s_prime).squeeze()
    a_max = q(s_prime).max(1)[1].unsqueeze(-1)
    r = torch.FloatTensor(r).unsqueeze(-1).to(device)
    done = torch.FloatTensor(done).unsqueeze(-1).to(device)
    with torch.no_grad():
        y = r + gamma * q_target(s_prime).gather(1, a_max) * done
    a = torch.tensor(a).unsqueeze(-1).to(device)
    q_value = torch.gather(q(s), dim=1, index=a.view(-1, 1).long())

    loss = F.smooth_l1_loss(q_value, y).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def copy_weights(q, q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)


# -----------------------------
# Main Function (Modified)
# -----------------------------
def main(env, q, q_target, optimizer, device):
    t = 0
    gamma = 0.99
    batch_size = 256

    N = 50000
    eps = 0.001
    memory = replay_memory(N)
    update_interval = 50
    print_interval = 10

    score_lst = []
    total_score = 0.0
    loss = 0.0
    start_time = time.perf_counter()

    csv_file = "training_scores.csv"
    
    # -----------------------------
    # RESUME: Determine start epoch
    # -----------------------------
    start_epoch = 0
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if not df.empty:
            start_epoch = int(df["epoch"].iloc[-1]) + print_interval
            score_lst = pickle.load(open("score.p", "rb"))  # resume scores if available
            print(f"Resuming from epoch {start_epoch}")

    # -----------------------------
    # CREATE CSV IF IT DOESN'T EXIST
    # -----------------------------
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_score", "avg_loss", "stage"])

    # -----------------------------
    # LOAD MODEL WEIGHTS IF AVAILABLE
    # -----------------------------
    if os.path.exists("mario_q.pth") and os.path.exists("mario_q_target.pth"):
        q.load_state_dict(torch.load("mario_q.pth"))
        q_target.load_state_dict(torch.load("mario_q_target.pth"))
        print("Loaded existing model weights.")

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    for k in range(start_epoch, 7000):
        s = arrange(env.reset())
        done = False

        while not done:
            if eps > np.random.rand():
                a = env.action_space.sample()
            else:
                if device == "cpu":
                    a = np.argmax(q(s).detach().numpy())
                else:
                    a = np.argmax(q(s).cpu().detach().numpy())
            s_prime, r, done, _ = env.step(a)
            s_prime = arrange(s_prime)
            total_score += r
            r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
            memory.push((s, float(r), int(a), s_prime, int(1 - done)))
            s = s_prime
            stage = env.unwrapped._stage
            if len(memory) > 2000:
                loss += train(q, q_target, memory, batch_size, gamma, optimizer, device)
                t += 1
            if t % update_interval == 0:
                copy_weights(q, q_target)
                torch.save(q.state_dict(), "mario_q.pth")
                torch.save(q_target.state_dict(), "mario_q_target.pth")

        if k % print_interval == 0:
            time_spent, start_time = (
                time.perf_counter() - start_time,
                time.perf_counter(),
            )
            print(
                "%s |Epoch : %d | score : %f | loss : %.2f | stage : %d | time spent: %f"
                % (
                    device,
                    k,
                    total_score / print_interval,
                    loss / print_interval,
                    stage,
                    time_spent,
                )
            )

            # -----------------------------
            # APPEND TRAINING RESULTS TO CSV
            # -----------------------------
            avg_score = total_score / print_interval
            avg_loss = loss / print_interval
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([k, avg_score, avg_loss, stage])
            # -----------------------------

            score_lst.append(avg_score)
            total_score = 0
            loss = 0.0
            pickle.dump(score_lst, open("score.p", "wb"))


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v3")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    q = model(n_frame, env.action_space.n, device).to(device)
    q_target = model(n_frame, env.action_space.n, device).to(device)
    optimizer = optim.Adam(q.parameters(), lr=0.0001)
    print(device)

    main(env, q, q_target, optimizer, device)
