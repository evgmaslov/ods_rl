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

#Init policy with plain distribution
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
    #Iterative value function's computation
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

#-----Additional functions to compare iteration methods
def value_iteration_step(v_values, gamma):
    q_values = get_q_values(v_values, gamma)
    new_v_values = init_v_values()
    for state in env.get_all_states():
        new_v_values[state] = 0
        max_q = 0
        for action in env.get_possible_actions(state):
            current_q = q_values[state][action]
            if current_q > max_q:
                max_q = current_q
        new_v_values[state] = max_q
    return new_v_values

def value_iteration(iter_n, gamma, log=False):
    policy = init_policy()
    v_values = init_v_values()
    for _ in range(iter_n):
        v_values = value_iteration_step(v_values, gamma)
        q_values = get_q_values(v_values, gamma)
        policy = policy_improvement(q_values)
        if log:
             mean_total_reward = policy_validation(policy, env_entering_n, action_n)
             wandb.log({"mean_total_reward":mean_total_reward})
    return policy

def policy_iteration(iter_n, eval_iter_n, gamma):
    policy = init_policy()
    for _ in range(iter_n):
        q_values = policy_evaluation(policy, gamma, eval_iter_n)
        policy = policy_improvement(q_values)
    return policy

#-----Definitely tasks-----
#Comment other tasks when running file on one task
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

wandb.login()
iter_n = 20
env_entering_n = 1000
action_n = 1000

#Value Iteration parameters search
#gamma
gamma_range = [0.9, 0.9999, 0.001]
config = {
    "task":"Parameters search",
    "description":"Searching optimal parameters for Valut Iteration",
    "gamma_range":gamma_range,
    "iter_n":iter_n,
    "env_entering_n":1000,
    "action_n":1000
}
run = wandb.init(project="ods_rl-frozen_lake", config=config, name="Parameters search_Run 2")

wandb.define_metric("gamma")
wandb.define_metric("Mean total reward", step_metric="gamma")

max_reward = 0
max_gamma = 0
for gamma in np.arange(*gamma_range):
    policy = value_iteration(iter_n, gamma)
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

#iter_n
iter_n = 20
gamma = 0.9902
config = {
    "task":"Parameters search",
    "description":"Searching optimal parameters for Valut Iteration",
    "gamma":gamma,
    "iter_n":iter_n,
    "env_entering_n":1000,
    "action_n":1000
}
run = wandb.init(project="ods_rl-frozen_lake", config=config, name="Parameters search_Run 3")

policy = value_iteration(iter_n, gamma, log=True)

#-----Comparising policy iteration and value iteration-----
pi_iter_n = 20
pi_gamma = 0.99979
pi_eval_iter_n = 20

vi_iter_n = 25
vi_gamma = 0.9902

env_entering_n = 1000
action_n = 1000


config = {
    "task":"Comparising policy iteration and value iteration",
    "description":"Try different gamma values with fixed number of iterations.",
    "iter_n":iter_n,
    "eval_iter_n":pi_eval_iter_n,
    "gamma":pi_gamma,
    "env_entering_n":1000,
    "action_n":1000
}
#run = wandb.init(project="ods_rl-frozen_lake", config=config, name="Run 6")



value_iteration_policy = value_iteration(vi_iter_n, vi_gamma)
value_mean_reward = policy_validation(value_iteration_policy, env_entering_n, action_n)

policy_iteration_policy = policy_iteration(pi_iter_n, pi_eval_iter_n, pi_gamma)
policy_mean_reward = policy_validation(policy_iteration_policy, env_entering_n, action_n)

print("Mean total reward for value iteration: {0}".format(value_mean_reward))
print("Mean total reward for policy iteration: {0}".format(policy_mean_reward))