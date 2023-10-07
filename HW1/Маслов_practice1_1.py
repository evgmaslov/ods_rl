import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
class CrossEntropyAgent():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)

    def get_trajectory(self, env, max_len=200, visualize=False, print_reward=False):
        trajectory = {'states': [], 'actions': [], 'rewards': []}

        obs = env.reset()
        state = obs
        if print_reward:
            print("test started")
        for _ in range(max_len):

            trajectory['states'].append(state)

            action = self.get_action(state)
            trajectory['actions'].append(action)

            obs, reward, done, _ = env.step(action)
            trajectory['rewards'].append(reward)
            if print_reward:
                print(reward)
            state = obs

            if visualize:
                time.sleep(0.5)
                env.render()

            if done:
                break
        return trajectory
    def copy(self):
        new_agent = CrossEntropyAgent(self.state_n, self.action_n)
        new_agent.model = self.model.copy()
        return new_agent

class Trainer():
    def __init__(self, env, state_n, action_n):
        self.env = env
        self.state_n = state_n
        self.action_n = action_n
        self.max_reward = -1000000
        self.best_agent = None
        self.best_params = {"q": None, "iteration_n": None, "trajectory_n": None}
    def train(self, agent, q_param, iteration_n, trajectory_n, visualize=False):
        rewards = []
        print("training with parameters: q: {0}, iter_n: {1}, trajectory_n: {2}".format(q_param, iteration_n, trajectory_n))
        for iteration in range(iteration_n):
            # policy evaluation
            trajectories = [agent.get_trajectory(self.env) for _ in range(trajectory_n)]
            total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
            rewards.append(np.mean(total_rewards))
            if visualize:
                clear_output()
                plt.plot(range(iteration + 1), rewards)
                plt.show()

            # policy improvement
            quantile = np.quantile(total_rewards, q_param)
            elite_trajectories = []
            for trajectory in trajectories:
                total_reward = np.sum(trajectory['rewards'])
                if total_reward > quantile:
                    elite_trajectories.append(trajectory)

            new_model = np.zeros((self.state_n, self.action_n))
            for trajectory in elite_trajectories:
                for state, action in zip(trajectory['states'], trajectory['actions']):
                    new_model[state][action] += 1

            for state in range(self.state_n):
                if np.sum(new_model[state]) > 0:
                    new_model[state] /= np.sum(new_model[state])
                else:
                    new_model[state] = agent.model[state].copy()

            agent.model = new_model
        if max(rewards) > self.max_reward:
            self.max_reward = max(rewards)
            self.best_agent = agent.copy()
            self.best_params["q"] = q_param
            self.best_params["iteration_n"] = iteration_n
            self.best_params["trajectory_n"] = trajectory_n
        return rewards
    def grid_search(self, agent, q_params, iter_params, tr_params, visualize=False, save_visualize="training.png"):
        q_params[1] += q_params[2]
        iter_params[1] += iter_params[2]
        tr_params[1] += tr_params[2]
        fig, ax = plt.subplots()
        ax.set_title("Train")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        for q in np.arange(*q_params):
            for iteration in np.arange(*iter_params):
                for tr in np.arange(*tr_params):
                    new_agent = agent.copy()
                    rewards = self.train(new_agent, q, iteration, tr)
                    print("max reward: {0}".format(max(rewards)))

                    if max(rewards) > self.max_reward:
                        self.max_reward = max(rewards)
                        self.best_agent = new_agent
                        self.best_params["q"] = q
                        self.best_params["iteration_n"] = iteration
                        self.best_params["trajectory_n"] = tr

                    if visualize:
                        clear_output()
                        label = "trajectory_n: {2}, max reward: {3}".format(q, iteration, tr, round(max(rewards), 3))
                        ax.plot(range(len(rewards)), rewards, label=label)
                        ax.legend()
                        plt.savefig(save_visualize)

env = gym.make('Taxi-v3')
state_n = 500
action_n = 6

agent = CrossEntropyAgent(state_n, action_n)
q_param = 0.9
iteration_n = 50
trajectory_n = 300
trainer = Trainer(env, state_n, action_n)
trainer.grid_search(agent, [0.7, 0.7, 0.1], [7, 7, 1], [5000, 5000, 1], visualize=True, save_visualize="training_1_1.png")
agent = trainer.best_agent
trajectory = agent.get_trajectory(env, max_len=100, visualize=True)
print('total reward:', sum(trajectory['rewards']))
print('model:')
print(agent.model)

