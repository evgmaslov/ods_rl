from Frozen_Lake import FrozenLakeEnv
import numpy as np
import time
import wandb

#-----Env initialization-----
env = FrozenLakeEnv()

#-----Functions-----
def get_q_values(v_values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                q_values[state][action] += env.get_transition_prob(state, action, next_state) * env.get_reward(state, action, next_state)
                q_values[state][action] += gamma * env.get_transition_prob(state, action, next_state) * v_values[next_state]
    return q_values

def init_policy():
	policy = {}
	for state in env.get_all_states():
		policy[state] = {}
		for action in env.get_possible_actions(state):
			policy[state][action] = 1/len(env.get_possible_actions(state))
	return policy

def init_v_values():
	v_values = {}
	for state in env.get_all_states():
		v_values[state] = 0
	return v_values

def policy_evaluation_step(v_values, policy, gamma):
    q_values = get_q_values(v_values, gamma)
    new_v_values = init_v_values()
    for state in env.get_all_states():
        new_v_values[state] = 0
        for action in env.get_possible_actions(state):
            new_v_values[state] += policy[state][action] * q_values[state][action]
    return new_v_values

def policy_evaluation(policy, gamma, eval_iter_n):
    v_values = init_v_values()
    for _ in range(eval_iter_n):
        v_values = policy_evaluation_step(v_values, policy, gamma)
    q_values = get_q_values(v_values, gamma)
    return q_values
	
def policy_improvement(q_values):
	policy = {}
	for state in env.get_all_states():
		policy[state] = {}
		argmax_action = None
		max_q_value = float("-inf")
		for action in env.get_possible_actions(state):
			policy[state][action] = 0
			if q_values[state][action] > max_q_value:
				argmax_action = action
				max_q_value = q_values[state][action]
		policy[state][argmax_action] = 1
	return policy

#-----Gamma validation-----
iter_n = 20
eval_iter_n = 20
gamma_range = [0.999, 0.99999, 0.0001]
env_entering_n = 1000
action_n = 1000

wandb.login()
config = {
    "task":"Gamma validation",
    "description":"Try different gamma values with fixed number of iterations.",
    "iter_n":iter_n,
    "eval_iter_n":eval_iter_n,
    "gamma_range":gamma_range,
    "env_entering_n":1000,
    "action_n":1000
}
run = wandb.init(project="ods_rl-frozen_lake", config=config, name="Run 3")

def policy_training(iter_n, eval_iter_n, gamma):
    policy = init_policy()
    for _ in range(iter_n):
        q_values = policy_evaluation(policy, gamma, eval_iter_n)
        policy = policy_improvement(q_values)
    return policy

def policy_validation(policy, env_entering_n, action_n):
    total_rewards = []
    for _ in range(env_entering_n):
        total_reward = 0
        state = env.reset()
        for _ in range(action_n):
            action = np.random.choice(env.get_possible_actions(state), p=list(policy[state].values()))
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if done:
                break
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

wandb.define_metric("gamma")
wandb.define_metric("Mean total reward", step_metric="gamma")

max_reward = 0
max_gamma = 0
for gamma in np.arange(*gamma_range):
    policy = policy_training(iter_n, eval_iter_n, gamma)
    mean_total_reward = policy_validation(policy, env_entering_n, action_n)
    if mean_total_reward > max_reward:
         max_gamma = gamma
         max_reward = mean_total_reward
    log_dict = {
        "gamma": gamma,
        "Mean total reward": mean_total_reward,
    }
    wandb.log(log_dict)
print("Max gamma: {0}".format(max_gamma))