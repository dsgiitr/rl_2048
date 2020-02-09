# Q Learning Agent to Play 2048
Implementation of deep Q-network to play the game 2048 using Keras.

## Deep Q-Learning
Q-Learning is a reinforcement learning algorithm that seeks to find the best action to take given the current state. The expected reward of a action at that step is known as the Q-Value. It is stored in a table for each state, future state tuple. For most environments keeping track of the number of possible states and all possible combinations of these state is extremely difficult. Hence instead of storing these Q-Values we approximate them using Neural networks, this is known as a Deep Q-Network.
Read more at-
- www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/
- https://arxiv.org/abs/1312.5602


## Project Description 
For calculating the Q Values we used a Neural Network, rather than just showing the current state to the network, the next 4 possible states(left, right, up, down) were also shown. This intuition was inspired from Monte Carlo Tree Search estimation where game is played till the end to determine the Q-Values. Instead of using a normal Q Network, a Double Q Network was used one for predicting Q values and other for predicting actions. This is done to try and reduce the large overestimations of action values which result form a positive bias introduced in Q Learning. For data preprocessing log2 normalisation, training was done using the Bellman's Equation. The policy used was Epsilon greedy, to allow exploration the value of epsilon was annealed down by 5%. 

## Results

![](https://github.com/dsgiitr/rl_2048/blob/master/Max_tile.png) 

Fig: Max tile obtained on each game as NN training proceeds. Its visible that the model is able to learn the strategy within 600 Epochs as the number of tiles with 512 is much more towards the end.

![](https://github.com/dsgiitr/rl_2048/blob/master/monotonicity.png)

Also from observation the model was able to learn the common heuristic amongst player to keep the maximum numbered tile at one corner and surround it with monotonically increasing tiles. This helps in combining the tiles in a series. 

## Repository Structure
- DQN_Agent_2048.ipynb-Main notebook to train/test the DQN, also contains the definitions of the deep learning models
- Game_Board.py-Module with the game logic of 2048 (uses OPENAI Gym interface)
- DQN_Network is the Architecture of the Q Network.
- trained_model.model-Model trained for 1000 epochs.

## Installing and Running
This project requires:
- Python (Python 3.6)
- Numpy
- TensorFlow
- Keras
- Gym (OpenAI Gym)
- Matplotlib

Once everything is installed open the DQN_Agent_2048.ipynb file on a jupyter notebook and run the cells.

## Future Work
This work is in the very elementry stage and we'd like to improve upon the following:
- Other Networks like Actor Critic
- Convolution Network in the Network architecture for Agent
- Showing the subsequent 4 States of all the immideate next state so as to get better predictions(4 +4*4=16 states in total)
- Run the game for more iterations (1,000,000)

## References
1. https://github.com/SergioIommi/DQN-2048
2. https://github.com/berkay-dincer/2048-ai-mcts
3. https://towardsdatascience.com/welcome-to-deep-reinforcement-learning-part-1-dqn-c3cab4d41b6b
4. http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
5. https://papers.nips.cc/paper/3964-double-q-learning
