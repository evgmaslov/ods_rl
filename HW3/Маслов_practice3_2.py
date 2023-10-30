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

#This function is modified with using values from the last step
def policy_evaluation(policy, gamma, eval_iter_n, last_v_values, use_last_values=True):
    if use_last_values:
        v_values = last_v_values
    else:
        v_values = init_v_values()
    for _ in range(eval_iter_n):
        v_values = policy_evaluation_step(v_values, policy, gamma)
    q_values = get_q_values(v_values, gamma)
    return q_values, v_values
	
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

#-----Trying to use last step values-----
iter_n = 20
eval_iter_n = 20
gamma = 0.99979
env_entering_n = 1000
action_n = 1000

wandb.login()
config = {
    "task":"Comparison of different values using strategies",
    "description":"Try different gamma values with fixed number of iterations.",
    "iter_n":iter_n,
    "eval_iter_n":eval_iter_n,
    "gamma":gamma,
    "env_entering_n":1000,
    "action_n":1000
}
run = wandb.init(project="ods_rl-frozen_lake", config=config, name="Run 5")

#This function is modified with using values from the last step
def policy_training(iter_n, eval_iter_n, gamma, use_last_values):
    policy = init_policy()
    v_values = init_v_values()
    rewards = []
    for _ in range(iter_n):
        q_values, v_values = policy_evaluation(policy, gamma, eval_iter_n, v_values, use_last_values=use_last_values)
        policy = policy_improvement(q_values)
        mean_total_reward = policy_validation(policy, env_entering_n, action_n)
        wandb_title = "Default strategy reward"
        if use_last_values:
             wandb_title = "Using last values strategy reward"
        wandb.log({wandb_title:mean_total_reward})
        rewards.append(mean_total_reward)
    return policy, rewards

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

policy, default_rewards = policy_training(iter_n, eval_iter_n, gamma, use_last_values=False)
policy, last_values_rewards = policy_training(iter_n, eval_iter_n, gamma, use_last_values=True)

print("Max default strategy reward: {0}".format(max(default_rewards)))
print("Max last values strategy reward: {0}".format(max(last_values_rewards)))