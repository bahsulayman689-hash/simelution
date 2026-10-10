"""Multi-agent version: several agents share the grid, block each other, learn independently."""
import random
from sim_env import MAP, ACTIONS, QAgent


class MultiGridWorld:
    def __init__(self, n_agents=3, layout=MAP, max_steps=200):
        self.grid = [list(r) for r in layout]
        self.n, self.max_steps, self.n_actions = n_agents, max_steps, len(ACTIONS)
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == "S":
                    s = (r, c)
        near = [(s[0] + dr, s[1] + dc) for dr, dc in [(0, 1), (1, 0), (0, 2), (2, 0)]]
        self.starts = ([s] + [p for p in near if self.grid[p[0]][p[1]] == "."])[:n_agents]
        assert len(self.starts) == n_agents, "too many agents for this map"
        self.reset()

    def reset(self):
        self.pos = list(self.starts)
        self.has_key = [False] * self.n
        self.done = [False] * self.n
        self.steps = 0
        return [self._obs(i) for i in range(self.n)]

    def _obs(self, i):
        # position, key flag, and which of the 4 neighbours hold another active agent
        r, c = self.pos[i]
        others = {p for j, p in enumerate(self.pos) if j != i and not self.done[j]}
        near = sum(1 << k for k, (dr, dc) in ACTIONS.items() if (r + dr, c + dc) in others)
        return (r, c, int(self.has_key[i]), near)

    def step(self, actions):
        self.steps += 1
        rewards, infos = [0.0] * self.n, [""] * self.n
        for i in random.sample(range(self.n), self.n):  # random order = fair conflicts
            if self.done[i]:
                continue
            dr, dc = ACTIONS[actions[i]]
            nr, nc = self.pos[i][0] + dr, self.pos[i][1] + dc
            cell = self.grid[nr][nc]
            occupied = {p for j, p in enumerate(self.pos) if j != i and not self.done[j]}
            rewards[i] = -0.01
            if cell == "#" or (cell == "D" and not self.has_key[i]):
                rewards[i], infos[i] = -0.05, "blocked"
            elif (nr, nc) in occupied:
                rewards[i], infos[i] = -0.1, "collision"
            else:
                self.pos[i] = (nr, nc)
                if cell == "K" and not self.has_key[i]:
                    self.has_key[i], rewards[i], infos[i] = True, 0.5, "picked up key"
                elif cell == "T":
                    rewards[i], self.done[i], infos[i] = -1.0, True, "fell in trap"
                elif cell == "G":
                    rewards[i], self.done[i], infos[i] = 1.0, True, "reached goal"
        if self.steps >= self.max_steps:
            for i in range(self.n):
                if not self.done[i]:
                    self.done[i], infos[i] = True, "timeout"
        return [self._obs(i) for i in range(self.n)], rewards, list(self.done), infos


def train_multi(env, agents, episodes=3000, report_every=500):
    goals = 0
    for ep in range(1, episodes + 1):
        obs = env.reset()
        while not all(env.done):
            active = [not d for d in env.done]
            acts = [agents[i].act(obs[i]) for i in range(env.n)]
            obs2, rew, _, info = env.step(acts)
            for i in range(env.n):
                if active[i]:
                    agents[i].learn(obs[i], acts[i], rew[i], obs2[i], env.done[i])
            obs = obs2
        goals += sum(1 for p, d in zip(env.pos, env.done) if env.grid[p[0]][p[1]] == "G")
        if ep % report_every == 0:
            print(f"episode {ep:5d} | agents reaching goal {goals / (ep * env.n):.2f}")


if __name__ == "__main__":
    random.seed(0)
    env = MultiGridWorld(n_agents=3)
    agents = [QAgent(env.n_actions) for _ in range(env.n)]
    train_multi(env, agents)
