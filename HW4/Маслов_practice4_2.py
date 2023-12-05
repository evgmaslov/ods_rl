import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
import torch
from torch import nn

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

def get_dce_action(self, config):
    device = self.config["device"]
    state = torch.FloatTensor(config["state"]).to(device)
    eps = config["eps"]

    logits = self.model(state)
    mean = 0
    std = eps
    noise = torch.tensor(np.random.normal(mean, std, logits.size()), dtype=torch.float).to(device)
    action_prob = self.config["activation"](logits+noise).detach().cpu().numpy()
    action = np.random.choice(self.config["action_n"], p=action_prob)
    return action

def get_trajectory(env, agent, agent_config, max_len=200):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()[0]
    state = obs
    for _ in range(max_len):

        trajectory['states'].append(state)

        agent_config["state"] = state
        action = agent.get_action(agent_config)
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

class DiscreteEnv():
    def __init__(self, env):
        self.env = env
        self.n_actions = 4
        self.discretization = [30, 30, 5, 5, 30, 5, 2, 2]
        self.obs_high = [1.5, 1.5, 5., 5., 3.14, 5., 1, 1]
        self.obs_low = [-1.5, -1.5, -5., -5., -3.14, -5., 0, 0]
        self.linspaces = []
        for i in range(len(self.obs_high)):
            lin = np.linspace(self.obs_high[i], self.obs_low[i], self.discretization[i])
            self.linspaces.append(lin)
        self.states = {}
        self.n_states = 0
        states = self.get_all_states_from(0)
        for state in states:
            state_hash = self.get_state_hash(state)
            self.states[state_hash] = self.n_states
            self.n_states += 1
    def get_all_states_from(self, ind):
        all_states = []
        counter = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(self.discretization[ind]):
            if ind == len(counter) - 1:
                state = counter.copy()
                all_states.append(state)
                counter[ind] += 1
            else:
                states = self.get_all_states_from(ind+1)
                for state in states:
                    new_state = state.copy()
                    new_state[ind] = i
                    all_states.append(new_state)
        return all_states
    def get_state_hash(self, state):
        state_hash = "_".join([str(item) for item in state])
        return state_hash
    def get_d_state(self, state):
        d_state_arr = []
        for ind, item in enumerate(state):
            i = (np.abs(self.linspaces[ind] - item)).argmin()
            d_state_arr.append(i)
        d_state = self.states[self.get_state_hash(d_state_arr)]
        return d_state
    def reset(self):
        state = self.env.reset()[0]
        d_state = self.get_d_state(state)
        return d_state
    def step(self, action):
        state, reward, done, some1, some2 = self.env.step(action)
        d_state = self.get_d_state(state)
        return d_state, reward, done, some1, some2

def DeepCrossEntropy(env, agent, config):
        
    q_param = config["q_param"]
    iteration_n = config["iteration_n"]
    trajectory_n = config["trajectory_n"]
    trajectory_len = config["trajectory_len"]
    device = config["device"]
    loss_fn = config["loss_fn"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]

    rewards = []
    
    eps = 1
    for iteration in tqdm(range(iteration_n), desc="Cross entropy training"):
        # policy evaluation
        agent_config = {"eps":eps, "device":device}
        trajectories = [get_trajectory(env, agent, agent_config, max_len=trajectory_len) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        rewards.extend(total_rewards)

        # policy improvement
        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['rewards'])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        if len(elite_trajectories) > 0:

            elite_states = []
            elite_actions = []
            for trajectory in elite_trajectories:
                elite_states.extend(trajectory['states'])
                elite_actions.extend(trajectory['actions'])
            elite_states = torch.FloatTensor(elite_states).to(device)
            elite_actions = torch.LongTensor(elite_actions).to(device)
            
            loss = loss_fn(agent.model(elite_states), elite_actions)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            eps = 1/(iteration+1)

        if iteration == iteration_n-1:
            rewards.extend(total_rewards)
    return rewards

def MonteCarlo(env, agent, config):

    episode_n = config["episode_n"]
    trajectory_len = config["trajectory_len"]
    gamma = config["gamma"]
    state_n = config["state_n"]
    action_n = config["action_n"]

    total_rewards = []
    
    
    counter = np.zeros((state_n, action_n))
    
    for episode in tqdm(range(episode_n), desc="Monte-Carlo training"):
        epsilon = 1 - episode / episode_n
        trajectory = {'states': [], 'actions': [], 'rewards': []}
        
        state = env.reset()
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
        
        state = env.reset()
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
        
        state = env.reset()
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

env = gym.make('LunarLander-v2')
state_n = 8
action_n = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

#Deep cross entropy
dce_agent_config = {
    "action_n":action_n,
    "state_n":state_n,
    "activation":nn.Softmax(),
    "device":device,
}
dce_agent_model = nn.Sequential(
            nn.Linear(state_n, 64), 
            nn.ReLU(), 
            nn.Linear(64, action_n)
        ).to(device)
dce_agent = Agent(
    model = dce_agent_model,
    config=dce_agent_config,
)
dce_agent.get_action = get_dce_action.__get__(dce_agent, Agent)

dce_config = {
    "q_param":0.8,
    "iteration_n":100,
    "trajectory_n":200,
    "trajectory_len":500,
    "device":device,
    "loss_fn":nn.CrossEntropyLoss(),
    "optimizer":torch.optim.AdamW(dce_agent.model.parameters(), lr=1e-2),
    "scheduler":None
}
#dce_rewards = DeepCrossEntropy(env, dce_agent, dce_config)

#Monte-Carlo
d_env = DiscreteEnv(env)

mc_agent = create_q_agent(d_env.n_states, d_env.n_actions)
mc_config = {
    "action_n":d_env.n_actions,
    "state_n":d_env.n_states,
    "episode_n":5000,
    "trajectory_len":1000,
    "gamma":0.99
}
mc_rewards = MonteCarlo(d_env, mc_agent, mc_config)

#SARSA
sarsa_agent = create_q_agent(d_env.n_states, d_env.n_actions)
sarsa_config = {
    "episode_n":1000,
    "trajectory_len":5000,
    "gamma":0.99,
    "alpha":0.1,
}
sarsa_rewards = SARSA(d_env, sarsa_agent, sarsa_config)

#Q-Learning
ql_agent = create_q_agent(d_env.n_states, d_env.n_actions)
ql_config = {
    "episode_n":5000,
    "trajectory_len":1000,
    "gamma":0.99,
    "alpha":0.1,
}
ql_rewards = Q_Learning(d_env, ql_agent, ql_config)

fig, ax = plt.subplots(figsize=(15, 5))
#ax.plot(np.arange(len(dce_rewards)), dce_rewards, label="Deep Cross Entropy")
ax.plot(np.arange(len(mc_rewards)), mc_rewards, label="Monte-Carlo")
ax.plot(np.arange(len(sarsa_rewards)), sarsa_rewards, label="SARSA")
ax.plot(np.arange(len(ql_rewards)), ql_rewards, label="Q-Learning")
ax.legend()

plt.savefig("Task_2.png")
