[//]: # (Image References)

[image1]:
https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif 
"Trained Agent"

# Project 1: Navigation

### Project Details

In this project, You can train an agent to navigate and collect bananas in a large, sqaure world. 


![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. 

The discrete action space has 4 dimensions.

- **`0`** - move forward
- **`1`** - move backward
- **`2`** - turn left
- **`3`** - turn right

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


### Getting Started

To run this project, You need several python packages, Unity ML-Agents Toolkit and the environment.

- numpy(>=1.11)
- pytorch(>=0.4)
- matplotlib(>=1.11)

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
2. Clone the udacity nanodegree repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies. This is for installing [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) and all the needed python packages.
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```
3. Download the unity environment from one of the links below. In this case you will download the Banana collector environment.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

4. Place the file in the data folder, and uzip the file.

### Instructions

There are 4 main elements in this project. 

- Report.ipynd
- model.py
- agent.py
- model_checkpoint.pth (soft_update_checkpoint.pth)

Report.ipynd includes simple summary of the algorithm and codes for training the agent, visualizing the rewards graphs and running the agent. In this project, I experimented two cases. One is the agent with vanilla target network update rule and the other is with soft update rule. The vanilla update rule which was suggested by the original DQN paper didn't work well. So I applied the soft update rule proposed by [Lillicrap et al.](http://arxiv.org/abs/1509.02971) You can try experiments by running the cells in this Report.ipynd file.


You can modify the network via model.py. You can change learning algorithm and hyperparmeters in agent.py. To train the agent, run the scripts in Repory.ipynd. 

model_checkpoint.pth is parameters of the agent who achieved the goal of the task. Instead of training, You can use this checkpoint directly to see how the agent behave and interact with the environemnt. To see how the agent behave, You can run the cell in Report.ipynd

### References

- Minh et al., [Human-level control through deep reinforcement learning](http://dx.doi.org/10.1038/nature14236) 
- Lillicrap et al., [Continuous control with deep reinforcement learning](http://arxiv.org/abs/1509.02971)

### License

This project is covered under the [MIT License.](./LICENSE)





