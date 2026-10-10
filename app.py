"""Streamlit viewer. Run: streamlit run app.py"""
import time
import random
import streamlit as st
from sim_env import GridWorld, QAgent, train
from multi_agent import MultiGridWorld, train_multi

st.set_page_config(page_title="Agent Simulation", layout="centered")
st.title("Agent Simulation")

EMOJI = {"#": "⬛", ".": "⬜", "S": "⬜", "T": "🕳️", "K": "🔑", "D": "🚪", "G": "🏁"}
AGENTS = ["🤖", "🦊", "🐸", "🐙"]


def grid_html(grid, positions, keys):
    rows = []
    for r, row in enumerate(grid):
        line = ""
        for c, ch in enumerate(row):
            if (r, c) in positions:
                line += AGENTS[positions.index((r, c))]
            elif ch in "KD" and all(keys):
                line += "⬜"
            else:
                line += EMOJI[ch]
        rows.append(line)
    return f"<div style='font-size:30px;line-height:1.2'>{'<br>'.join(rows)}</div>"


def play(mode, agents):
    """Run one greedy episode, return (grid, frames, summary)."""
    if mode == "Multi-agent":
        env = MultiGridWorld(n_agents=len(agents))
        obs, frames = env.reset(), [(list(env.pos), list(env.has_key))]
        while not all(env.done):
            obs, _, _, _ = env.step([a.act(o) for a, o in zip(agents, obs)])  # keeps eps=0.05 so agents can break deadlocks
            frames.append((list(env.pos), list(env.has_key)))
        goals = sum(env.grid[p[0]][p[1]] == "G" for p in env.pos)
        return env.grid, frames, f"{goals}/{env.n} agents reached the goal in {env.steps} steps"
    env = GridWorld()
    s, done, info = env.reset(), False, ""
    frames = [([env.pos], [env.has_key])]
    while not done:
        s, _, done, info = env.step(agents[0].act(s, greedy=True))
        frames.append(([env.pos], [env.has_key]))
    return env.grid, frames, f"{info} in {env.steps} steps"


mode = st.sidebar.radio("Agent", ["Q-learning", "DQN (PyTorch)", "Multi-agent"])
episodes = st.sidebar.slider("Training episodes", 200, 5000, 1500, 100)
n_agents = st.sidebar.slider("Agents (multi-agent)", 2, 4, 3) if mode == "Multi-agent" else 1
speed = st.sidebar.slider("Playback speed (seconds per step)", 0.02, 0.5, 0.12)

store = st.session_state.setdefault("trained", {})

if st.button("Train"):
    random.seed(0)
    with st.spinner("Training..."):
        if mode == "Q-learning":
            env = GridWorld()
            agent = QAgent(env.n_actions)
            train(env, agent, episodes, report_every=10**9)
            store[mode] = [agent]
        elif mode == "DQN (PyTorch)":
            try:
                from dqn_agent import DQNAgent
            except ImportError:
                st.error("PyTorch is not installed. Run: pip install torch")
                st.stop()
            env = GridWorld()
            agent = DQNAgent(len(env.grid), len(env.grid[0]), env.n_actions)
            train(env, agent, min(episodes, 1000), report_every=10**9)
            store[mode] = [agent]
        else:
            env = MultiGridWorld(n_agents=n_agents)
            agents = [QAgent(env.n_actions) for _ in range(n_agents)]
            train_multi(env, agents, episodes, report_every=10**9)
            store[mode] = agents
    st.success("Training done.")

if st.button("Play episode"):
    if mode not in store:
        st.warning("Train this agent first.")
    else:
        grid, frames, summary = play(mode, store[mode])
        box = st.empty()
        for positions, keys in frames:
            box.markdown(grid_html(grid, positions, keys), unsafe_allow_html=True)
            time.sleep(speed)
        st.info(summary)