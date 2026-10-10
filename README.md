# Agent Simulation

A small grid-world simulation where AI agents learn to find a key, open a door and reach a goal while avoiding a trap. It includes three kinds of agent and a Streamlit page that animates them.

**Live demo:** [add your streamlit.app link here]

## The world

```
##########
#S..#....#
#.#.#.##.#
#.#...#K.#
#.####.#.#
#..T...#.#
##.###.#.#
#..#....D#
#....#.#G#
##########
```

`S` start, `K` key, `D` door (needs the key), `G` goal, `T` trap, `#` wall.

| Event | Reward |
|---|---|
| Reach the goal | +1.0 |
| Pick up the key | +0.5 |
| Fall in the trap | -1.0 |
| Collide with another agent | -0.1 |
| Walk into a wall or locked door | -0.05 |
| Every step | -0.01 |

Each episode ends at the goal, in the trap, or after 200 steps. The environment follows the Gym style: `reset()` returns an observation and `step(action)` returns `(obs, reward, done, info)`.

## Agents

- **Q-learning** (`sim_env.py`): a tabular agent. It needs no dependencies and reaches the goal about 99% of the time.
- **DQN** (`dqn_agent.py`): a PyTorch deep Q-network with a replay buffer and a target network. It has the same `act` and `learn` interface as the Q-learning agent.
- **Multi-agent** (`multi_agent.py`): 2 to 4 agents share the map and block each other. Each learns on its own and can sense neighbouring agents.

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

In the sidebar, pick an agent, press **Train**, then press **Play episode** to watch it move.

You can also run each part from the terminal:

```
python sim_env.py        # Q-learning, prints training progress and a demo run
python multi_agent.py    # multi-agent training
python dqn_agent.py      # DQN training (needs PyTorch), saves dqn.pt
```

## Files

| File | Purpose |
|---|---|
| `sim_env.py` | Environment and Q-learning agent |
| `multi_agent.py` | Multi-agent environment and training loop |
| `dqn_agent.py` | PyTorch DQN agent |
| `app.py` | Streamlit viewer |
| `requirements.txt` | Dependencies |

## Make it your own

- Edit the `MAP` list at the top of `sim_env.py` to build a new world.
- Plug in your own agent by writing a class with `act(state, greedy=False)` and `learn(s, a, r, s2, done)`.

## Deploy

Push the folder to GitHub, then create an app at [share.streamlit.io](https://share.streamlit.io) with `app.py` as the main file. If the install fails on size, use the CPU-only PyTorch line in `requirements.txt` or remove PyTorch. The DQN option then shows an install message and the other two agents still work.

## Author

Sulayman Bah. GitHub: github.com/bahsulayman689-hash
