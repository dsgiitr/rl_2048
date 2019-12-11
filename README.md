# Q Learning Agent to Play 2048
Implementation of deep Q-network to play the game 2048 using Keras.

## Deep Q-Learning
Q-Learning is a reinforcement learning algorithm that seeks to find the best action to take given the current state. The expected reward of a action at that step is known as the Q-Value. It is stored in a table for each state, future state tuple. For most environments keeping track of the number of possible states and all possible combinations of these state is extremely difficult. Hence instead of storing these Q-Values we approximate them using Neural networks, this is known as a Deep Q-Network.
Read more at-
- www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/
- https://arxiv.org/abs/1312.5602


## Project Description 
For calculating the Q Values we used a Neural Network, rather than just showing the current state to the network, the next 4 possible states(left, right, up, down) were also shown. This intuition was inspired from Monte Carlo Tree Search where game is played till the end to determine the Q-Values. 
For data preprocessing log2 normalisation, training was done using the Bellman's Equation. The policy used was Epsilon greedy, to allow exploration the value of epsilon was annealed down by 5%. 

## Repository Structure
- DQN_Agent_2048.ipynb-Main notebook to train/test the DQN, also contains the definitions of the deep learning models
- Game_Board.py-Module with the game logic of 2048 (uses OPENAI Gym interface)
- trained_model.model-Model trained for 1000 epochs.

## References
1. https://github.com/SergioIommi/DQN-2048
2. https://github.com/berkay-dincer/2048-ai-mcts
3. https://towardsdatascience.com/welcome-to-deep-reinforcement-learning-part-1-dqn-c3cab4d41b6b
4. http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
