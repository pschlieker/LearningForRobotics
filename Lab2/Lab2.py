import numpy as np

x = 0.25
y = 0.25
gamma = .9

#Store Transition matrices
actions = 3
states = 4

#Action, Current state, Next state
T = np.zeros((actions, states, states))
T[0, 1, 1] = 1-x
T[0, 1, 3] = x
T[0, 2, 0] = 1-y
T[0, 2, 3] = y
T[0, 3, 0] = 1
T[1, 0, 1] = 1
T[2, 0, 2] = 1

#Rewards
R = np.array([0,0,1,10])

#Initialize value vector
V = np.zeros(states)

#Initialize policy vector
P = np.full(states, -1)

#Value Iteration
while True:
	V_new = np.zeros(states)

	#Update Value for each state
	for cs in range(states):
		V_new[cs] = R[cs] #Reward for state

		#Calculate Value of each action
		V_actions = np.zeros(actions)
		for a in range(actions):
			for ns in range(states):
				V_actions[a] += T[a,cs,ns] * V[ns]

		#Select best action and store value
		V_new[cs] += gamma * max(V_actions)
		P[cs] = np.where(V_actions == max(V_actions))[0][0]

	#Check if convergence has been reached
	dif = np.sum(V - V_new)
	if abs(dif) < 0.001 :
		break
	V = V_new

print("V: "+str(V))
print("Pi: "+str(P))