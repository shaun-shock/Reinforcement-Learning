import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, m):
        self.m = m
        self.mean = 0
        self.N = 0
        self.q = 10

    def choose(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x
    def update_Optimistic(self, x, alpha):
        self.N += 1
        self.q = (1-alpha)*self.q + alpha*x
    def update_UCB(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x
    

def run_experiment(means, eps, N):
    num_actions = len(means)
    actions = [Bandit(m) for m in means]
    data = np.empty(N)
    optimal_action_counts = np.zeros(N)  # Track the count of selecting the optimal action
    average_rewards = []  # Store average rewards at each time step

    optimal_action_index = np.argmax(means)  # Index of the optimal action

    for i in range(N):
        # epsilon-greedy
        p = np.random.random()
        if p < eps:
            j = np.random.choice(num_actions)
        else:
            j = np.argmax([a.mean for a in actions])
        
        # Track the count of selecting the optimal action at this time step
        if j == optimal_action_index:
            optimal_action_counts[i] = 1
        
        x = actions[j].choose()
        actions[j].update(x)
        data[i] = x

        # Calculate and store the average reward up to this point
        average_reward = np.mean(data[:i + 1])
        average_rewards.append(average_reward)

    # Calculate the percentage of times the optimal action was selected at each time step
    percent_optimal_action = np.cumsum(optimal_action_counts) / np.arange(1, N + 1) * 100

    return data, percent_optimal_action, average_rewards

def run_experiment_opti(means, N,alpha):
    num_actions = len(means)
    actions = [Bandit(m) for m in means]
    data = np.empty(N)
    optimal_action_counts = np.zeros(N)  # Track the count of selecting the optimal action
    average_rewards = []  # Store average rewards at each time step

    optimal_action_index = np.argmax(means)  # Index of the optimal action

    for i in range(N):
        p = np.random.random()
        j = np.argmax([a.q for a in actions])
        # Track the count of selecting the optimal action at this time step
        if j == optimal_action_index:
            optimal_action_counts[i] = 1
        
        x = actions[j].choose()
        actions[j].update_Optimistic(x,alpha)
        data[i] = x

        # Calculate and store the average reward up to this point
        average_reward = np.mean(data[:i + 1])
        average_rewards.append(average_reward)

    # Calculate the percentage of times the optimal action was selected at each time step
    percent_optimal_action = np.cumsum(optimal_action_counts) / np.arange(1, N + 1) * 100

    return data, percent_optimal_action, average_rewards

def run_experiment_UCB(means, N, c):
    selection_rate = np.zeros(len(means))
    num_actions = len(means)
    actions = [Bandit(m) for m in means]
    data = np.empty(N)
    average_rewards = []
    optimal_action_counts = np.zeros(N) 
    optimal_action_index = np.argmax(means) 
    maxpos=0
    for j in range(N):
        #print(selection_rate)
        for i in range(len(actions)):
            #maxpos=1
            if(selection_rate[i]==0):
                maxpos =  i
                break
            elif(actions[i].mean+ c * np.sqrt(np.log(j+1)/selection_rate[i]) > actions[maxpos].mean+ c * np.sqrt(np.log(j+1)/selection_rate[maxpos])):
                maxpos = i
        selection_rate[maxpos] = selection_rate[maxpos]+1
        
        x = actions[maxpos].choose()
        actions[maxpos].update_UCB(x)
        data[j] = x
        average_reward = np.mean(data[:j + 1])
        average_rewards.append(average_reward)
        if maxpos == optimal_action_index:
            optimal_action_counts[j] = 1

    # Calculate the percentage of times the optimal action was selected at each time step
    percent_optimal_action = np.cumsum(optimal_action_counts) / np.arange(1, N + 1) * 100

    return data, percent_optimal_action, average_rewards
        

class MDP:
    def __init__(self, num_states, num_actions, reward_function):
        self.num_states = num_states
        self.num_actions = num_actions
        self.reward_function = reward_function
        self.V = np.full(num_states, 0.5)  # Initialize the value function with 0.5 for all states
        self.V[0]=0
        self.V[6]=0
        self.rms_errors=[]

    def td_0(self, alpha, gamma, num_episodes):
        for episode in range(num_episodes):
            state = 3  # Initialize the starting state
            while state != 0 and state!=6 :
        # Choose an action
                action = state + np.random.choice([-1, 1])
        
        # Simulate the environment: Transition to the next state and observe reward
                next_state = action
                reward = self.reward_function[next_state]
        
        # Update the value function using TD(0) update rule
                self.V[state] += alpha * (reward + gamma * self.V[next_state] - self.V[state])
        
        # Move to the next state
                state = next_state
                
            rms_error = calculate_rms_error(np.array[0,1/6,2/6,3/6,4/6,5/6,0])
            self.rms_errors.append(rms_error)


    def td_0_avg(self, alpha, gamma, num_episodes):
        N=np.zeros(self.num_states)
        for episode in range(num_episodes):
            state = 3  # Initialize the starting state
            while state != 0 and state!=6 :
        # Choose an action
                action = state + np.random.choice([-1, 1])
                N[state]+=1
        
        # Simulate the environment: Transition to the next state and observe reward
                next_state = action
                reward = self.reward_function[next_state]
        
        # Update the value function using TD(0) update rule
                self.V[state] += (1/N[state]) * (reward + gamma * self.V[next_state] - self.V[state])
        
        # Move to the next state
                state = next_state



def calculate_rms_error(self, true_values):
    return np.sqrt(np.mean((self.V - true_values) ** 2))
    



if __name__ == '__main__':
    action_means = [1.0, 2.0, 3.0,4.0,3.50, 3.71, 4.80, -2.25, -2.51, -2.27]  # Reward_function



"""  Please uncomment any of the snippets here to perform a simulation of any of the methods, you can tinker the values of epsilon, alpha and c as well for further analysis """ 



#     epsilon_1 = 0.1
#     epsilon_2 = 0.2
#     epsilon_05 = 0.05
#     num_steps = 1000
    
#     c_2, percent_optimal_2, avg_rewards_2 = run_experiment_opti(action_means,num_steps, 0.2)
#     c_1, percent_optimal_1, avg_rewards_1 = run_experiment_UCB(action_means,num_steps, 0.1)
#     c_05, percent_optimal_05, avg_rewards_05 = run_experiment_UCB(action_means,num_steps, 0.05)
#     c_05, percent_optimal_05, avg_rewards_05 = run_experiment_opti(action_means, num_steps,0.1)
#     c_05, percent_optimal_05, avg_rewards_05 = run_experiment_UCB(action_means, num_steps, 2)
#     run_experiment_UCB(action_means, num_steps, 2)




#     plt.figure(figsize=(12, 8))
#     plt.plot(avg_rewards_2, label='alpha=0.2')
#     plt.plot(avg_rewards_1, label='alpha=0.1')
#     plt.plot(avg_rewards_05, label='alpha=0.05')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Average Reward')
#     plt.legend()
#     plt.show()

#     # Plot the percentage of times the optimal action is selected at each time step
#     plt.figure(figsize=(12, 8))
#     plt.plot(percent_optimal_2, label='alpha=0.2')
#     plt.plot(percent_optimal_1, label='alpha=0.1')
#     plt.plot(percent_optimal_05, label='alpha=0.05')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Percentage of Optimal Action')
#     plt.legend()
#     plt.show()

#     # Initial values for the value function
#     initial_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

# # Create an instance of MDP with the initial values
#     mdp0 = MDP(7, 6, initial_values)
#     mdp1 = MDP(7, 6, initial_values)
#     mdp2 = MDP(7, 6, initial_values)
#     mdp3 = MDP(7, 6, initial_values)


# # Call the TD(0) method with your desired parameters
#     gamma = 1
#     num_episodes = 10000
#     mdp0.td_0_avg(0.2, gamma, 10)
#     mdp1.td_0(0.2, gamma, 10)
#     mdp2.td_0(0.2, gamma, 10)
#     mdp3.td_0(0.2, gamma, 100)

# # Plot the estimated values for each state
#     plt.figure(figsize=(8, 6))

#     plt.plot(mdp0.rms_errors)
#     plt.xlabel('Episode')
#     plt.ylabel('RMS Error')
#     plt.title('RMS Error Over Episodes')
#     plt.show()
    
#     plt.plot(np.arange(5), mdp0.V[1:6], marker='o', linestyle='-', label='ep=0')
#     plt.plot(np.arange(5), mdp1.V[1:6], marker='o', linestyle='-', label='ep=1')
#     plt.plot(np.arange(5), mdp2.V[1:6], marker='o', linestyle='-', label='ep=10')
#     plt.plot(np.arange(5), mdp3.V[1:6], marker='o', linestyle='-', label='ep=100')
#     plt.legend()
#     plt.xlabel("State")
#     plt.ylabel("Estimated Value")
#     plt.title("Estimated Values for Each State")
#     plt.grid(True)
#     plt.show()

    
