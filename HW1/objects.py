import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm


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
    def __init__(self, env, state_n, action_n, stochastic_env=False, m=1):
        self.env = env
        self.state_n = state_n
        self.action_n = action_n
        self.max_reward = -1000000
        self.best_agent = None
        self.best_params = {"q": None, "iteration_n": None, "trajectory_n": None}
        self.stochastic_env = stochastic_env
        self.m = m
        if self.m == 1:
            self.m = action_n

    def train(self, agent, q_param, iteration_n, trajectory_n, lmbd=0.0, visualize=False, smoothing="laplace"):
        rewards = []
        print(
            "training with parameters: q: {0}, iter_n: {1}, trajectory_n: {2}, lambda: {3}".format(q_param, iteration_n,
                                                                                                   trajectory_n, lmbd))
        for iteration in range(iteration_n):
            total_rewards = []
            if self.stochastic_env:
                stochastic_policy = agent.model.copy()
                trajectories = []
                for i in tqdm(range(self.m), desc="Iteration {0}".format(iteration)):
                    determ_policy = np.zeros((self.state_n, self.action_n))
                    for state in range(self.state_n):
                        action = np.random.choice(np.arange(self.action_n), p=stochastic_policy[state])
                        determ_policy[state][action] = 1
                    agent.model = determ_policy
                    trs = [agent.get_trajectory(self.env) for _ in range(trajectory_n)]
                    trajectories = trajectories + trs
                    local_reward = np.mean([np.sum(trajectory['rewards']) for trajectory in trs])
                    total_rewards += [local_reward for _ in trs]
                rewards.append(np.mean(total_rewards))
                agent.model = stochastic_policy

            else:
                # policy evaluation
                trajectories = [agent.get_trajectory(self.env) for _ in tqdm(range(trajectory_n), desc="Iteration {0}, Sampling trajectories".format(iteration))]
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
            if lmbd == 0.0:
                for state in range(self.state_n):
                    if np.sum(new_model[state]) > 0:
                        new_model[state] /= np.sum(new_model[state])
                    else:
                        new_model[state] = agent.model[state].copy()
                agent.model = new_model
            elif smoothing == "laplace":
                for state in range(self.state_n):
                    if np.sum(new_model[state]) > 0:
                        s = np.sum(new_model[state])
                        new_model[state] += lmbd
                        new_model[state] /= (s + self.action_n * lmbd)
                    else:
                        new_model[state] += lmbd
                        new_model[state] /= (self.action_n * lmbd)
                agent.model = new_model
            elif smoothing == "policy":
                for state in range(self.state_n):
                    if np.sum(new_model[state]) > 0:
                        new_model[state] /= (np.sum(new_model[state]))
                    else:
                        new_model[state] = agent.model[state].copy()

                agent.model = new_model * lmbd + agent.model * (1 - lmbd)
        if max(rewards) > self.max_reward:
            self.max_reward = max(rewards)
            self.best_agent = agent.copy()
            self.best_params["q"] = q_param
            self.best_params["iteration_n"] = iteration_n
            self.best_params["trajectory_n"] = trajectory_n
        return rewards

    def grid_search(self, agent, q_params, iter_params, tr_params, lmbd_params=[0, 0, 0], visualize=False,
                    save_visualize="training.png", smoothing="laplace"):
        q_params[1] += q_params[2]
        iter_params[1] += iter_params[2]
        tr_params[1] += tr_params[2]
        lmbd_params[1] += lmbd_params[2]
        fig, ax = plt.subplots()
        ax.set_title("Train")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        for q in np.arange(*q_params):
            for iteration in np.arange(*iter_params):
                for tr in np.arange(*tr_params):
                    for lmbd in np.arange(*lmbd_params):
                        new_agent = agent.copy()
                        rewards = self.train(new_agent, q, iteration, tr, lmbd, smoothing=smoothing)
                        print("max reward: {0}".format(max(rewards)))

                        if max(rewards) > self.max_reward:
                            self.max_reward = max(rewards)
                            self.best_agent = new_agent
                            self.best_params["q"] = q
                            self.best_params["iteration_n"] = iteration
                            self.best_params["trajectory_n"] = tr

                        if visualize:
                            clear_output()
                            label = "q: {0} trajectory_n: {1}, lambda: {2}, max reward: {3}".format(q, tr, lmbd,
                                                                                round(max(rewards), 3))
                            if self.stochastic_env:
                                label += (", m = " + str(self.m))
                            ax.plot(range(len(rewards)), rewards, label=label)
                            ax.legend()
                            plt.savefig(save_visualize)