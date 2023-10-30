from Frozen_Lake import FrozenLakeEnv
import numpy as np
import time

env = FrozenLakeEnv()

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
        max_q = 0
        for action in env.get_possible_actions(state):
            current_q = q_values[state][action]
            if current_q > max_q:
                max_q = current_q
        new_v_values[state] = max_q
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

iter_n = 20
eval_iter_n = 20
gamma = 0.9

policy = init_policy()
for _ in range(iter_n):
	q_values = policy_evaluation(policy, gamma, eval_iter_n)
	policy = policy_improvement(q_values)

total_reward = 0
state = env.reset()
for _ in range(1000):
	action = np.random.choice(env.get_possible_actions(state), p=list(policy[state].values()))
	state, reward, done, _ = env.step(action)
	total_reward += reward
	env.render()
	time.sleep(0.5)
	if done:
		break
print(total_reward)