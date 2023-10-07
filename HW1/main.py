import gym
from objects import CrossEntropyAgent, Trainer

env = gym.make('Taxi-v3')
state_n = 500
action_n = 6

agent = CrossEntropyAgent(state_n, action_n)
q_param = 0.9
iteration_n = 50
trajectory_n = 300
trainer = Trainer(env, state_n, action_n, stochastic_env=True, m=50)
trainer.grid_search(agent, [0.3, 0.3, 0.1], [20, 20, 1], [300, 300, 1], [0.8, 0.8, 0.1],
                    visualize=True, save_visualize="training_3_10.png", smoothing="policy")
agent = trainer.best_agent
n_tests = 1
for _ in range(n_tests):
    trajectory = agent.get_trajectory(env, max_len=100, visualize=True)
print('total reward:', sum(trajectory['rewards']))
print('model:')
print(agent.model)