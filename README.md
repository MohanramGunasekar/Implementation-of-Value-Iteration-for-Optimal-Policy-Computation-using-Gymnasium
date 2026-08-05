# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement
Implement the Value Iteration algorithm on the FrozenLake-v1 environment using the Gymnasium library. The objective is to iteratively compute the optimal value function and derive the optimal policy that maximizes the expected cumulative reward.

## Software Requirements
<p>Python 3.x</p>
Gymnasium
NumPy
Matplotlib
Jupyter Notebook / Google Colab / VS Code


## Environment Description
FrozenLake-v1 is a grid-world reinforcement learning environment where an agent starts from the Start (S) state and must reach the Goal (G) while avoiding Holes (H).

The environment consists of:

S – Start State
F – Frozen Surface (Safe)
H – Hole (Terminal State)
G – Goal State

Possible Actions:

Left (0)
Down (1)
Right (2)
Up (3)

Rewards:

Goal = +1
All other states = 0



## MDP Representation
An MDP is represented by the tuple
```text
(S,A,P,R,γ)
```
where

S = Set of states
A = Set of actions
P = Transition probability
R = Reward function
γ = Discount factor

## Theory
Value Iteration is a Dynamic Programming algorithm used to determine the optimal value function for every state.

It repeatedly updates each state's value using the Bellman Optimality Equation until convergence.

Bellman Optimality Equation:
```text
V(s)=amax​s′∑​P(s′∣s,a)[R+γV(s′)]
```
After obtaining the optimal value function, the optimal policy is extracted by selecting the action with the highest expected return for each state.

## Algorithm
Initialize all state values to zero.
For every state:
Compute the value for each possible action.
Update the state value using the maximum action value.
Continue updating until the maximum change in values is less than a threshold.
Extract the optimal policy by choosing the action with the highest value for every state.
Display the optimal value function and policy.


## Python Program

```python

# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------
def value_iteration(env, gamma=0.99, theta=1e-8):

    nS = env.observation_space.n
    nA = env.action_space.n

    V = np.zeros(nS)
    iteration = 0

    while True:
        delta = 0

        for s in range(nS):
            v = V[s]

            action_values = np.zeros(nA)

            for a in range(nA):
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            V[s] = np.max(action_values)
            delta = max(delta, abs(v - V[s]))

        iteration += 1

        if delta < theta:
            break

    # Extract Optimal Policy
    policy = np.zeros(nS, dtype=int)

    for s in range(nS):
        action_values = np.zeros(nA)

        for a in range(nA):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                action_values[a] += prob * (
                    reward + gamma * V[next_state] * (not done)
                )

        policy[s] = np.argmax(action_values)

    return V, policy, iteration

```

---

## Output

```text
env_desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]
```
<img width="351" height="350" alt="image" src="https://github.com/user-attachments/assets/f526d9c8-212b-492d-9176-3e30465e1901" />

---

## Result
```text
The Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The optimal state-value function and the corresponding optimal policy were obtained using the Bellman Optimality Equation.

```
---

## Inference
```text
env_desc = [
    "FFFS",
    "FHFH",
    "FFFH",
    "GHFF"
]
```
<img width="315" height="348" alt="image" src="https://github.com/user-attachments/assets/5640ea07-7b4e-4e54-9ceb-5b61fc86fff6" />

---

