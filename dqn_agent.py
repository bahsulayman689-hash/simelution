"""PyTorch DQN agent for the grid world. Same interface as QAgent (act / learn / eps).
Run: python dqn_agent.py   (needs: pip install torch)"""
import random
from collections import deque
import torch
import torch.nn as nn
from sim_env import GridWorld, train, demo


class QNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, n_out))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, h, w, n_actions, lr=1e-3, gamma=0.97, eps=1.0, eps_min=0.05,
                 eps_decay=0.999, batch=64, sync=200, buffer=20000):
        self.h, self.w, self.n = h, w, n_actions
        self.n_in = h * w + 1  # one-hot position + key flag
        self.q, self.target = QNet(self.n_in, n_actions), QNet(self.n_in, n_actions)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = deque(maxlen=buffer)
        self.gamma, self.eps, self.eps_min, self.eps_decay = gamma, eps, eps_min, eps_decay
        self.batch, self.sync, self.steps = batch, sync, 0

    def encode(self, obs):
        v = torch.zeros(self.n_in)
        v[obs[0] * self.w + obs[1]] = 1.0
        v[-1] = float(obs[2])
        return v

    def act(self, obs, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n)
        with torch.no_grad():
            return int(self.q(self.encode(obs)).argmax())

    def learn(self, s, a, r, s2, done):
        self.buf.append((self.encode(s), a, r, self.encode(s2), float(done)))
        self.steps += 1
        if len(self.buf) >= self.batch:
            S, A, R, S2, D = zip(*random.sample(self.buf, self.batch))
            S, S2 = torch.stack(S), torch.stack(S2)
            A = torch.tensor(A)
            R, D = torch.tensor(R, dtype=torch.float32), torch.tensor(D)
            qsa = self.q(S).gather(1, A.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                tgt = R + self.gamma * (1 - D) * self.target(S2).max(1).values
            loss = nn.functional.smooth_l1_loss(qsa, tgt)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
            self.opt.step()
        if self.steps % self.sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path="dqn.pt"):
        torch.save(self.q.state_dict(), path)

    def load(self, path="dqn.pt"):
        self.q.load_state_dict(torch.load(path))
        self.target.load_state_dict(self.q.state_dict())


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    env = GridWorld(seed=0)
    agent = DQNAgent(len(env.grid), len(env.grid[0]), env.n_actions)
    train(env, agent, episodes=800, report_every=100)
    agent.save()
    demo(env, agent)
