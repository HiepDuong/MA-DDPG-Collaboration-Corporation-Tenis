# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg_agent import Agent, ReplayBuffer
from utils.utils import agent_batch_dim_swap
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size

class MADDPG:
    def __init__(self, state_size, action_size, num_agents, random_seed):
        super(MADDPG, self).__init__()

        self.memory = ReplayBuffer(
            action_size = action_size,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            seed= random_seed
        )
        self.maddpg_agent = [Agent(state_size, random_seed, self.memory, BATCH_SIZE, action_size) for _ in range(num_agents)]

        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        
        self.t_step = 0

                    
    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()

    def act(self, env_states, noise):
#          return np.concatenate([
#             agent.act(state).reshape(1, 2) 
#             for agent, state in zip(self.maddpg_agent, env_states)
#         ])
        actions = [agent.act(state, noise) for agent, state in zip(self.maddpg_agent, env_states)]
        return actions

    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        for i in range(self.num_agents):
            self.maddpg_agent[i].memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
            self.maddpg_agent[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i])
#             Save experience for the other agents.

            
    def save(self):
        """
        Save network parameters.
        """
        for i, agent in enumerate(self.maddpg_agent):
            torch.save(agent.actor_local.state_dict(), f"checkpoint_actor_local_{i}.pth")
            torch.save(agent.actor_target.state_dict(), f"checkpoint_actor_target_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"checkpoint_critic_local_{i}.pth")
            torch.save(agent.critic_target.state_dict(), f"checkpoint_critic_target_{i}.pth")
            
    def load(self):
        """
        Load network parameters.
        """
        for i, agent in enumerate(self.maddpg_agent):
            agent.actor_local.load_state_dict(torch.load(f"checkpoint_actor_local_{i}.pth"))
            agent.actor_target.load_state_dict(torch.load(f"checkpoint_actor_target_{i}.pth"))
            agent.critic_local.load_state_dict(torch.load(f"checkpoint_critic_local_{i}.pth"))
            agent.critic_target.load_state_dict(torch.load(f"checkpoint_critic_target_{i}.pth"))
            
            
            




