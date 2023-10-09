import torch
from torch import nn 
import numpy as np
import gym
import matplotlib.pyplot as plt
"""from pyvirtualdisplay import Display
from IPython import display as ipythondisplay"""
from gym.wrappers.record_video import RecordVideo


class CEM(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, self.action_n)
        )
        
        self.softmax = nn.Softmax()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, _input):
        return self.network(_input.to(device)) 
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        logits = self.forward(state)
        action_prob = self.softmax(logits).detach().cpu().numpy()
        action = np.random.choice(self.action_n, p=action_prob)
        return action
    
    def update_policy(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        elite_states = torch.FloatTensor(elite_states).to(device)
        elite_actions = torch.LongTensor(elite_actions).to(device)
        
        loss = self.loss(self.forward(elite_states), elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        
def get_trajectory(env, agent, trajectory_len, visualize=False):
    trajectory = {'states':[], 'actions': [], 'total_reward': 0}
    
    state = env.reset()[0]
    trajectory['states'].append(state)
    for i in range(trajectory_len):
        
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        
        state, reward, done, some1, some2 = env.step(action)
        trajectory['total_reward'] += reward
        
        if done:
            break
            
        if visualize:
            env.render()
        
        if i < trajectory_len - 1:
            trajectory['states'].append(state)
            
    return trajectory

def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param) 
    return [trajectory for trajectory in trajectories if trajectory['total_reward'] > quantile]

def wrap_env(env):
    env = RecordVideo(env, './video',  episode_trigger = lambda episode_number: True)
    return env

print("Started!")
"""display = Display(visible=0, size=(1400, 900))
display.start()
print("Display started!")"""
env = gym.make('LunarLander-v2', render_mode="rgb_array")
state_dim = 8
action_n = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Agent initialization")
agent = CEM(state_dim, action_n).to(device)
episode_n = 50
trajectory_n = 20
trajectory_len = 500
q_param = 0.8

rewards = []
for episode in range(episode_n):
    trajectories = [get_trajectory(env, agent, trajectory_len) for _ in range(trajectory_n)]
    
    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
    print(f'episode: {episode}, mean_total_reward = {mean_total_reward}')
    rewards.append(mean_total_reward)
    
    elite_trajectories = get_elite_trajectories(trajectories, q_param)
    
    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)
    
fig, ax = plt.subplots()
ax.set_title("Train")
ax.set_xlabel("Iteration")
ax.set_ylabel("Reward")
ax.plot(range(len(rewards)), rewards)
ax.legend()
plt.savefig("Task1_1.png")

env = wrap_env(env)
get_trajectory(env, agent, trajectory_len, visualize=True)