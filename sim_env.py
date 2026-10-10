"""
Simulation environment for an agent: a grid world with walls, traps, a key and a door.
Gym-style API: reset() -> obs, step(action) -> (obs, reward, done, info).
Pure Python, no dependencies. Run: python sim_env.py
"""
import random

MAP = [
    "##########",
    "#S..#....#",
    "#.#.#.##.#",
    "#.#...#K.#",
    "#.####.#.#",
    "#..T...#.#",
    "##.###.#.#",
    "#..#....D#",
    "#....#.#G#",
    "##########",
]
# S start, K key, D door (needs key), G goal, T trap, # wall

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
ACTION_NAMES = ["up", "down", "left", "right"]


class GridWorld:
    def __init__(self, layout=MAP, max_steps=200, seed=None):
        self.grid = [list(row) for row in layout]
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.n_actions = len(ACTIONS)
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == "S":
                    self.start = (r, c)
        self.reset()

    def reset(self):
        self.pos = self.start
        self.has_key = False
        self.steps = 0
        self.key_taken = False
        return self._obs()

    def _obs(self):
        return (self.pos[0], self.pos[1], int(self.has_key))

    def step(self, action):
        self.steps += 1
        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        cell = self.grid[nr][nc]
        reward, done, info = -0.01, False, ""

        if cell == "#" or (cell == "D" and not self.has_key):
            reward, info = -0.05, "blocked"
        else:
            self.pos = (nr, nc)
            if cell == "K" and not self.has_key:
                self.has_key, reward, info = True, 0.5, "picked up key"
            elif cell == "T":
                reward, done, info = -1.0, True, "fell in trap"
            elif cell == "G":
                reward, done, info = 1.0, True, "reached goal"

        if self.steps >= self.max_steps and not done:
            done, info = True, "timeout"
        return self._obs(), reward, done, info

    def render(self):
        out = []
        for r, row in enumerate(self.grid):
            line = ""
            for c, ch in enumerate(row):
                if (r, c) == self.pos:
                    line += "A"
                elif ch == "K" and self.has_key:
                    line += "."
                elif ch == "D" and self.has_key:
                    line += "."
                else:
                    line += ch
            out.append(line)
        return "\n".join(out)


class QAgent:
    """Tabular Q-learning agent."""

    def __init__(self, n_actions, alpha=0.2, gamma=0.97, eps=1.0, eps_min=0.05, eps_decay=0.9995):
        self.q = {}
        self.n = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay

    def _qs(self, s):
        return self.q.setdefault(s, [0.0] * self.n)

    def act(self, s, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n)
        qs = self._qs(s)
        best = max(qs)
        return random.choice([a for a, v in enumerate(qs) if v == best])

    def learn(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * max(self._qs(s2))
        self._qs(s)[a] += self.alpha * (target - self._qs(s)[a])
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


def train(env, agent, episodes=4000, report_every=500):
    wins = 0
    for ep in range(1, episodes + 1):
        s, done, info = env.reset(), False, ""
        while not done:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.learn(s, a, r, s2, done)
            s = s2
        wins += info == "reached goal"
        if ep % report_every == 0:
            print(f"episode {ep:5d} | goal rate (all) {wins / ep:.2f} | epsilon {agent.eps:.2f}")


def demo(env, agent):
    s, done = env.reset(), False
    print("\nGreedy run:\n" + env.render())
    while not done:
        s, r, done, info = env.step(agent.act(s, greedy=True))
    print(f"\nFinished in {env.steps} steps: {info or 'ok'}\n" + env.render())


if __name__ == "__main__":
    random.seed(0)
    env = GridWorld(seed=0)
    agent = QAgent(env.n_actions)
    train(env, agent)
    demo(env, agent)