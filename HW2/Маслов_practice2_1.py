import wandb
import torch
from torch import nn 
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym.wrappers.record_video import RecordVideo


class CEM(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, self.action_n)
        )
        
        self.softmax = nn.Softmax()
        self.optimizer = None
        self.scheduler = None
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, _input):
        return self.network(_input.to(device)) 
    
    def get_action(self, state, eps):
        state = torch.FloatTensor(state).to(device)
        logits = self.forward(state)
        mean = 0
        std = eps
        noise = torch.tensor(np.random.normal(mean, std, logits.size()), dtype=torch.float).to(device)
        action_prob = self.softmax(logits+noise).detach().cpu().numpy()
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
        if scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()
        
        
def get_trajectory(env, agent, trajectory_len, eps, visualize=False):
    trajectory = {'states':[], 'actions': [], 'total_reward': 0}
    
    state = env.reset()[0]
    trajectory['states'].append(state)
    for i in range(trajectory_len):
        
        action = agent.get_action(state, eps)
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

env = gym.make('LunarLander-v2', render_mode="rgb_array")
state_dim = 8
action_n = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

episode_n = 100
agent = CEM(state_dim, action_n).to(device)
optim = torch.optim.AdamW(agent.parameters(), lr=1e-2)
scheduler = None
agent.optimizer = optim
agent.scheduler = scheduler
trajectory_n = 200
trajectory_len = 500
q_param = 0.8

wandb.login()
config = {
    "trajectory_n":trajectory_n,
    "epochs":episode_n,
    "trajectory_len":trajectory_len,
    "q_param":q_param,
    "optim":optim,
    "scheduler":scheduler
}
run = wandb.init(project="ods_rl-lunar_lander", config=config, name="Run 10")
wandb.watch(agent, log_freq=100)

rewards = []
eps = 1
for episode in range(episode_n):
    trajectories = [get_trajectory(env, agent, trajectory_len, eps) for _ in range(trajectory_n)]
    
    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
    print(f'episode: {episode}, mean_total_reward = {mean_total_reward}')
    rewards.append(mean_total_reward)
    wandb.log({"mean_total_reward":mean_total_reward})
    
    elite_trajectories = get_elite_trajectories(trajectories, q_param)
    
    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)
        eps = 1/(episode+1)
    
fig, ax = plt.subplots()
ax.set_title("Train")
ax.set_xlabel("Iteration")
ax.set_ylabel("Reward")
ax.plot(range(len(rewards)), rewards)
ax.legend()
plt.savefig("Task1_5.png")

env = wrap_env(env)
get_trajectory(env, agent, trajectory_len, eps, visualize=False)
video_path = "/mnt/c/Все файлы/Курсы/ODS RL/video/rl-video-episode-0.mp4"
wandb.log({"video": wandb.Video(video_path, fps=4, format="gif")})
wandb.finish()