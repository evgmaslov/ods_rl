import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
import inspect
import math

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

def MonteCarlo(env, agent, config):

    episode_n = config["episode_n"]
    trajectory_len = config["trajectory_len"]
    gamma = config["gamma"]
    epsilons = config["epsilons"]

    assert len(epsilons) == episode_n

    total_rewards = []
    
    state_n = env.observation_space.n
    action_n = env.action_space.n
    counter = np.zeros((state_n, action_n))
    
    for episode in tqdm(range(episode_n), desc="Monte-Carlo training"):
        epsilon = epsilons[episode]
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

env = gym.make("Taxi-v3")
state_n = env.observation_space.n
action_n = env.action_space.n

#Monte-Carlo
mc_config = {
    "episode_n":1000,
    "trajectory_len":1000,
    "gamma":0.99,
}

strategies = [lambda episode: 1 - episode / mc_config["episode_n"],
              lambda episode: 1 - 0.2**((mc_config["episode_n"]-episode)/mc_config["episode_n"]),
              lambda episode: math.cos(math.pi*(episode+mc_config["episode_n"])/(2*mc_config["episode_n"]))+1,
              lambda episode: math.cos(math.pi*(episode+mc_config["episode_n"])/(2*mc_config["episode_n"]))/2+1,
              lambda episode: math.cos(math.pi*(episode+mc_config["episode_n"])/(2*mc_config["episode_n"]))/2+1-(2-episode/mc_config["episode_n"])/10,]
fig, ax = plt.subplots(1, len(strategies)+1, figsize=(30*(len(strategies)+1), 10))
for ind, st in enumerate(strategies):
    mc_config["eps_fn"] = st
    mc_config["epsilons"] = [mc_config["eps_fn"](episode) for episode in range(mc_config["episode_n"])]
    mc_agent = create_q_agent(state_n, action_n)
    mc_rewards = MonteCarlo(env, mc_agent, mc_config)

    funcString = str(inspect.getsourcelines(mc_config["eps_fn"])[0])
    #funcString = funcString.strip("['\\n']").split(" = ")[1]
    ax[0].plot(np.arange(len(mc_rewards)), mc_rewards, label=funcString)
    ax[ind+1].plot(np.arange(len(mc_rewards)), mc_rewards, label=funcString)
    ax[ind+1].legend()


plt.savefig("Task_3_2.png")
