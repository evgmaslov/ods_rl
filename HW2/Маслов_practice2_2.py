import wandb
wandb.login()
import torch
from torch import nn 
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym.wrappers.record_video import RecordVideo


class CEM(nn.Module):
    def __init__(self, state_dim, action_n, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, self.action_n),
        )
        
        self.activation = nn.Tanh()
        self.optimizer = None
        self.scheduler = None
        self.loss = nn.MSELoss()
        
    def forward(self, _input):
        out = self.network(_input.to(device))
        return out
    
    def get_action(self, state, eps):
        state = torch.FloatTensor(state).to(device)
        out = self.forward(state)
        mean = 0
        std = eps
        noise = torch.tensor(np.random.normal(mean, std, out.size()), dtype=torch.float).to(device)
        #action = np.array([-1.0]) if self.activation(out).item() < 0 else np.array([1.0])
        action = (self.activation(out)+noise).detach().cpu().numpy()
        return action
    
    def update_policy(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        elite_states = torch.FloatTensor(np.array(elite_states)).to(device)
        elite_actions = torch.FloatTensor(np.array(elite_actions)).to(device)
        
        loss = self.loss(self.forward(elite_states), elite_actions)
        loss.backward()
        self.optimizer.step()
        if scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()
        
        
def get_trajectory(env, agent, trajectory_len, eps, visualize=False):
    trajectory = {'states':[], 'actions': [], 'total_reward': 0, "done": False}
    
    state = env.reset()[0]
    trajectory['states'].append(state)
    for i in range(trajectory_len):
        
        action = agent.get_action(state, eps)
        trajectory['actions'].append(action)
        
        state, reward, done, some1, some2 = env.step(action)
        trajectory['total_reward'] += reward
        trajectory['done'] = done
        
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
    return [trajectory for trajectory in trajectories if (trajectory['total_reward'] > quantile)]

def wrap_env(env):
    env = RecordVideo(env, './video',  episode_trigger = lambda episode_number: True)
    return env

env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array")
state_dim = 2
action_n = 1
hidden_dim = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

episode_n = 500
agent = CEM(state_dim, action_n, hidden_dim).to(device)
optim = torch.optim.SGD(agent.parameters(), lr=0.01)
scheduler = None
loss = nn.L1Loss()
agent.optimizer = optim
agent.scheduler = scheduler
agent.loss = loss
trajectory_n = 1000
trajectory_len = 5000
q_param = 0.8


config = {
    "description":"Delete noise",
    "trajectory_n":trajectory_n,
    "epochs":episode_n,
    "trajectory_len":trajectory_len,
    "q_param":q_param,
    "optim":optim,
    "scheduler":scheduler,
    "hidden_dim":hidden_dim,
    "n_hidden":1,
    "loss":loss
}
run = wandb.init(project="ods_rl-MountainCarContinuous", config=config, name="Run 16")
wandb.watch(agent, log_freq=100)

rewards = []
eps = 0
rate=0.99
for episode in range(episode_n):
    trajectories = [get_trajectory(env, agent, trajectory_len, eps) for _ in range(trajectory_n)]
    
    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
    print(f'episode: {episode}, mean_total_reward = {mean_total_reward}')
    rewards.append(mean_total_reward)
    wandb.log({"mean_total_reward":mean_total_reward})
    
    elite_trajectories = get_elite_trajectories(trajectories, q_param)
    
    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)
        eps = eps*rate
    

fig, ax = plt.subplots()
ax.set_title("Train")
ax.set_xlabel("Iteration")
ax.set_ylabel("Reward")
ax.plot(range(len(rewards)), rewards)
ax.legend()
plt.savefig("Task2_4.png")

env = wrap_env(env)
get_trajectory(env, agent, trajectory_len, eps, visualize=False)
wandb.finish()