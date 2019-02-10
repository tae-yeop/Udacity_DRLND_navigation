import numpy as np
from collections import deque, namedtuple
import random

from model import Network

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                      
# Hyperparmeters
LR = 1e-3 # learning rate for optimizer
BATCH_SIZE = 128 # minibatch size
BUFFER_SIZE = int(1e4) # buffer size of replay memory
GAMMA = 0.99 # discount factor
TAU = 1e-3 # soft update
UP_FREQ = 4 # update frequency for the network

class Agent():
    """
    DQN Agent
    """
    def __init__(self, state_size, action_size, seed, soft_update, lr=LR, bf_size=BUFFER_SIZE, 
                 batch_size=BATCH_SIZE, update_freq=UP_FREQ, gamma=GAMMA, tau=1e-3):
        """
        Intialize agent
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.bf_size = bf_size
        self.soft_update = soft_update
        
        # Q-Networks
        self.qnetwork_local = Network(state_size, action_size,seed).to(device)
        self.qnetwork_target = Network(state_size, action_size, seed).to(device)
        # Optimizer
        self.optimizer = optim.Adam(params=self.qnetwork_local.parameters(), lr=lr)
        
        #Replay Memory
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.bf_size, self.batch_size, seed)
        
        # Initialize time step
        self.t_step = 0
        
        self.gamma = gamma
        self.tau = tau
        self.update_freq = update_freq
        
    def step(self, state, action, reward, next_state, done):
        """
        Save experience and do learning every self.update_freq

        Params
        =====
        states (numpy array) [state_size,]
        actions (numpy array) [action_size,]
        rewards (float list) 
        next_states (numpy array) [state_size,]
        dones (boolean list) 

        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Exectue the learning every update_freq
        self.t_step = (self.t_step + 1) % self.update_freq
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                # sample the experience
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
                
    def act(self, state, eps=0.0):
        """
        Select agents' actions to interact with the environment
        Params
        =====
        state (numpy array) [state_size,] : state vector
        eps (float) : epsilon value 
        
        Retruns
        =====
        Index of selected action (integer)
        """
        # convert state into Torch tensor [1,state_size] 
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection for exploration.
        if random.random() > eps: 
            # Choose best action
            return np.argmax(action_values.cpu().data.numpy()) # type: numpy.int64
        else: 
            # randomly choose action
            return random.choice(np.arange(self.action_size)) # type: numpy.int32
        
    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        
        Pramas
        ====
        experiences (Tupe of tensor) : states, actions, rewards, next_states, dones
        gamma (float) : discount ratio
        """
        states, actions, rewards, next_states, dones = experiences
        
        criterion = nn.MSELoss()
        q_estimate = self.qnetwork_local(states).gather(1, actions)
        
        # -----------Updatet local network----------------- #
        # [batch_size, action_size] -> [batch_size, 1]
        # Since the target network is updated manually, not by optimizer
        # Use detach() to ignore
        td_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1) 
        td_target = rewards + gamma*(td_target_next*(1-dones))
        
        loss = criterion(q_estimate, td_target)
        # Clear the graidents of model parameters
        self.optimizer.zero_grad()
        # Calucatlate the graidents
        loss.backward()
        # Update the network parameters
        self.optimizer.step()
        
        #update target network
        if self.soft_update == False:
            self.updatge_target(self.qnetwork_local, self.qnetwork_target)
        else:
            self.soft_update_target(self.qnetwork_local, self.qnetwork_target, self.tau)
        
    def updatge_target(self, local_model, target_model):
        """
        Vanilla update target network parameters
        
        Params
        ======
        local_model (PyTorch model)
        target_model (PyTorch model)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
   

    def soft_update_target(self, local_model, target_model, tau):
        """
        Soft update target network parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
        local_model (PyTorch model)
        target_model (PyTorch model)
        tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
            
class ReplayBuffer():
    def __init__(self, buf_size, batch_size, seed):
        """
        Params
        ======
        buf_size (int): size of memory
        batch_size (int): number of samples to be sampled
        seed (int): random seed
        """
        # When the replay buffer was full, the oldest sample needs to be discarded.
        # So deque is suitalbe data structure. 
        self.memory = deque(maxlen=buf_size)
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=["state", "action", "reward", 
                                                                "next_state", "done"])
        
    def __len__(self):
        """
        Return the size of memory
        """
        
        return len(self.memory)
    def add(self, state, action, reward, next_state, done):
        """
        Add the agent's experiences at eacy time to the memory
        """
        # Instantiate new experience with custom nemaedTuple
        e = self.experience(state, action, reward, next_state, done)
        # Add the tuple to the memory
        self.memory.append(e)
        
    def sample(self):
        """
        Draw a sample.
        Since the sample data is used by pytorch model, It needs to be converted to a torch Tensor.
        
        Returns
        ======
        A tuple of torch tensor. Each tenosr's outermost dimension is batch_size.
        """
        # list of sampled experience namedtuple of size of self.batch_size
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Use np.vstack() to make the first dimension is batch size.
        # states : [batch_size, state.shape]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # sicne action is used by torch.gather() during agent's learning step, 
        # It converts to long type.
        actions = torch.from_numpy(np.vstack([e.action for e in experiences 
                                              if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences 
                                              if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences 
                                                  if e is not None])).float().to(device)
        # dones is needed to calculated the Q-value. At terminal state(dones=1), 
        # the Q-value should be just latest rewards.
        dones = torch.from_numpy(np.vstack([e.done for e in experiences 
                                            if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)