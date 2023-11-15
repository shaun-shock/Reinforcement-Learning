# Reinforcement-Learning

This is a basic analysis I have done as part of my project on the various Update
functions used in Reinforcement Learning and the Markov Decision Process. 

# Part 1 : Analysis on Update Functions

I've included the following as part of my analysis :

1) Average Reward vs Time-Step of Execution of Agent
2) % of times the Optimal Reward was chosen vs Time-Step of Execution of Agent

   (Please Note : Comparison has been made over a period of 1000 time steps)

I have used three update functions as part of the analysis :
1) Epsilon-Greedy method (for more info, please check out : https://en.wikipedia.org/wiki/Multi-armed_bandit) :
    Analysis has been done for three different values of epsilon controlling the exploration of moves :
     a) 0.1
     b) 0.05
     c) 0.2
     
2) Optimistic Initial Values method (for more info, please check out : https://medium.com/@farbluestar/reinforcement-learning-part-03-355be5c7cae4) :
   Analysis has been done for three different values of alpha controlling the exploration of moves :
     a) 0.1
     b) 0.05
     c) 0.2
   
3) Upper Confidence Bound method (for more info, please check out : https://towardsdatascience.com/the-upper-confidence-bound-ucb-bandit-algorithm-c05c2bf4c13f ) :
      Analysis has been done for three different values of constant c controlling the exploration of moves :
     a) 0.5
     b) 2.0
     c) 3.0

4) Further, after finding the optimal values for the three methods above, a comparative analysis has been done for the three methods to compare the best method out of the
   three for epsilon = 0.5, alpha = 0.1 and c = 2.


   The report shows all the above findings and the source code provided shows the code which was used to run the simulations.



# Part 2 : Analysis on Markov Reward Process

I have also done a basic analysis of a basic Temporal Difference Learning Algorithm called Markov Reward or TD[0] process. In this analysis, I have used the following model for
my project where the leftmost and rightmost states are terminal states, and only the rightmost state has a reward of 1, the rest have a reward of 0. The aim is to see how the model estimates the average
reward of each state which is the expected reward from that state.


![image](https://github.com/shaun-shock/Reinforcement-Learning/assets/93643578/edaa440b-21a3-4750-88b6-b2446903d358)

The analysis has been done for various number of episodes( 0, 1 , 10 and 100). The findings have been shown in the report. The actual rewards for each state are as follows (from left to right) :


![image](https://github.com/shaun-shock/Reinforcement-Learning/assets/93643578/858e8595-db2d-4070-9ae2-7574d8c74f5e)

   The report shows all the above findings and the source code provided shows the code which was used to run the simulations.





   Really appreciate anyone taking their time to have a look at my project. I hope it was helpful. Thank you!
