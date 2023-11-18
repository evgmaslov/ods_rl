import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings

disable_warnings(InsecureRequestWarning)

class Agent():
    def __init__(self, model, config=None):
        self.model = model
        self.config = config
    def get_action(self, config):
        raise NotImplementedError("Getting action is not implemented")

def get_epsilon_greedy_action(self, config):
    policy = np.ones(self.config["action_n"]) * config["epsilon"] / self.config["action_n"]
    max_action = np.argmax(self.model[config["state"]])
    policy[max_action] += 1 - config["epsilon"]
    return np.random.choice(np.arange(self.config["action_n"]), p=policy)

def get_ce_action(self, config):
    """print(config["state"])
    print(type(config["state"]))"""
    action = np.random.choice(np.arange(self.config["action_n"]), p=self.model[config["state"]])
    return int(action)

def get_trajectory(env, agent, max_len=200):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()[0]
    state = obs
    for _ in range(max_len):

        trajectory['states'].append(state)

        action = agent.get_action(config={"state":state})
        trajectory['actions'].append(action)

        obs, reward, done, some1, some2 = env.step(action)
        trajectory['rewards'].append(reward)
        state = obs

        if done:
            break
    return trajectory

def create_q_agent(state_n, action_n):
    q_agent_config = {
        "action_n":action_n,
    }
    q_agent = Agent(
        model=np.zeros((state_n, action_n)),
        config=q_agent_config
    )
    q_agent.get_action = get_epsilon_greedy_action.__get__(q_agent, Agent)
    return q_agent

def CrossEntropy(env, agent, config):
        
    q_param = config["q_param"]
    iteration_n = config["iteration_n"]
    trajectory_n = config["trajectory_n"]

    rewards = []
    
    for iteration in tqdm(range(iteration_n), desc="Cross entropy training"):
        # policy evaluation
        trajectories = [get_trajectory(env, agent, max_len=200) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        rewards.extend(total_rewards)

        # policy improvement
        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['rewards'])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        new_model = np.zeros((agent.config["state_n"], agent.config["action_n"]))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(agent.config["state_n"]):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = agent.model[state].copy()

        agent.model = new_model
        if iteration == iteration_n-1:
            rewards.extend(total_rewards)
    return rewards

def MonteCarlo(env, agent, config):

    episode_n = config["episode_n"]
    trajectory_len = config["trajectory_len"]
    gamma = config["gamma"]

    total_rewards = []
    
    state_n = env.observation_space.n
    action_n = env.action_space.n
    counter = np.zeros((state_n, action_n))
    
    for episode in tqdm(range(episode_n), desc="Monte-Carlo training"):
        epsilon = 1 - episode / episode_n
        trajectory = {'states': [], 'actions': [], 'rewards': []}
        
        state = env.reset()[0]
        for _ in range(trajectory_len):
            trajectory['states'].append(state)
            
            action = agent.get_action({"state":state, "epsilon":epsilon})
            trajectory['actions'].append(action)
            
            state, reward, done, some1, some2 = env.step(action)
            trajectory['rewards'].append(reward)
            
            if done:
                break
                
        total_rewards.append(sum(trajectory['rewards']))
        
        real_trajectory_len = len(trajectory['rewards'])
        returns = np.zeros(real_trajectory_len + 1)
        for t in range(real_trajectory_len - 1, -1, -1):
            returns[t] = trajectory['rewards'][t] + gamma * returns[t + 1]
            
        for t in range(real_trajectory_len):
            state = trajectory['states'][t]
            action = trajectory['actions'][t]
            agent.model[state][action] += (returns[t] - agent.model[state][action]) / (1 + counter[state][action])
            counter[state][action] += 1
            
    return total_rewards

def SARSA(env, agent, config):

    episode_n = config["episode_n"]
    trajectory_len = config["trajectory_len"]
    gamma = config["gamma"]
    alpha = config["alpha"]

    total_rewards = []
    
    for episode in tqdm(range(episode_n), desc="SARSA training"):
        epsilon = 1 / (episode + 1)
        
        state = env.reset()[0]
        action = agent.get_action({"state":state, "epsilon":epsilon})
        total_reward = 0
        for _ in range(trajectory_len):
            next_state, reward, done, some1, some2 = env.step(action)
            next_action = agent.get_action({"state":next_state, "epsilon":epsilon})
            
            agent.model[state][action] += alpha * (reward + gamma * agent.model[next_state][next_action] - agent.model[state][action])
            
            state = next_state
            action = next_action
            
            total_reward += reward
            
            if done:
                break
        total_rewards.append(total_reward)

    return total_rewards

def Q_Learning(env, agent, config):

    episode_n = config["episode_n"]
    trajectory_len = config["trajectory_len"]
    gamma = config["gamma"]
    alpha = config["alpha"]

    total_rewards = []
    
    for episode in tqdm(range(episode_n), desc="Q-Learning training"):
        epsilon = 1 / (episode + 1)
        
        state = env.reset()[0]
        total_reward = 0
        for _ in range(trajectory_len):
            action = agent.get_action({"state":state, "epsilon":epsilon})
            next_state, reward, done, some1, some2 = env.step(action)

            agent.model[state][action] += alpha * (reward + gamma * np.max(agent.model[next_state]) - agent.model[state][action])
            
            state = next_state
            
            total_reward += reward
            
            if done:
                break
        total_rewards.append(total_reward)

    return total_rewards

env = gym.make("Taxi-v3")
state_n = env.observation_space.n
action_n = env.action_space.n

#Cross entropy
ce_agent_config = {
    "action_n":action_n,
    "state_n":state_n,
}
ce_agent = Agent(
    model = np.ones((state_n, action_n)) / action_n,
    config=ce_agent_config,
)
ce_agent.get_action = get_ce_action.__get__(ce_agent, Agent)
ce_config = {
    "q_param":0.9,
    "iteration_n":7,
    "trajectory_n":15000
}
ce_rewards = CrossEntropy(env, ce_agent, ce_config)

#Monte-Carlo
mc_agent = create_q_agent(state_n, action_n)
mc_config = {
    "episode_n":1000,
    "trajectory_len":1000,
    "gamma":0.99
}
mc_rewards = MonteCarlo(env, mc_agent, mc_config)

#SARSA
sarsa_agent = create_q_agent(state_n, action_n)
sarsa_config = {
    "episode_n":1000,
    "trajectory_len":1000,
    "gamma":0.99,
    "alpha":0.5,
}
sarsa_rewards = SARSA(env, sarsa_agent, sarsa_config)

#Q-Learning
ql_agent = create_q_agent(state_n, action_n)
ql_config = {
    "episode_n":1000,
    "trajectory_len":1000,
    "gamma":0.99,
    "alpha":0.5,
}
ql_rewards = Q_Learning(env, ql_agent, ql_config)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(np.arange(len(ce_rewards)), ce_rewards, label="CrossEntropy")
ax.plot(np.arange(len(mc_rewards)), mc_rewards, label="Monte-Carlo")
ax.plot(np.arange(len(sarsa_rewards)), sarsa_rewards, label="SARSA")
ax.plot(np.arange(len(ql_rewards)), ql_rewards, label="Q-Learning")
ax.legend()

plt.savefig("Task_1.png")
