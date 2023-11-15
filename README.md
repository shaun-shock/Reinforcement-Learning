# Reinforcement-Learning

This is a basic analysis I have done as part of my project on the various Update
functions used in Reinforcement Learning. I've included the following as part of my analysis :

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


   Really appreciate anyone taking their time to have a look at my project. I hope it was helpful. Thank you!
