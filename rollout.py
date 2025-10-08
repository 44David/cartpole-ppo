import torch
import numpy as np
import gymnasium as gym
import torch.optim as optim
from policy import CartPoleAgent
from value_function import ValueFunction
from ppo import ppo

def main():
    env = gym.make("CartPole-v1")
    
    agent = CartPoleAgent()
    critic = ValueFunction()

    policy_optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    vf_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
    
    n_timesteps = 2048
    
    cycles = 100
    
    observation, info = env.reset()
    for i in range(cycles):
        
        print(f"Rollout epoch {i}")
        total_timesteps = 0
        
        trajectories = {
            'observations': torch.zeros(n_timesteps, 4),
            'actions': torch.zeros(n_timesteps),
            'log_probs': torch.zeros(n_timesteps),
            'rewards': torch.zeros(n_timesteps),
            'values': torch.zeros(n_timesteps),
            'episode_over': torch.zeros(n_timesteps),
        }
        
        
        total_reward = 0
        for _ in range(n_timesteps):
            trajectories["observations"][total_timesteps] = torch.Tensor(observation).clone()
            
            out = agent(torch.Tensor(observation))
            value_estimate = critic(torch.Tensor(observation))
            
            action_dist = torch.distributions.Categorical(out)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            trajectories["actions"][total_timesteps] = action.detach()
            trajectories["log_probs"][total_timesteps] = log_prob    
            trajectories["values"][total_timesteps] = value_estimate.detach()
            
            # observation: (cart position, cart velocity, pole angle, pole angular velocity)
            observation, reward, terminated, truncated, info = env.step(action.item())
            
            trajectories["rewards"][total_timesteps] = reward    
            trajectories["episode_over"][total_timesteps] = 1 if terminated or truncated else 0
                
            total_timesteps += 1
            
            total_reward += reward
            
            if terminated or truncated:
                print(f"#### Total Reward {total_reward}")
                total_reward = 0
                observation, info = env.reset()
                
                
        agent, critic = ppo(trajectories, agent, critic, n_timesteps, policy_optimizer, vf_optimizer)
        
    env.close()
    
    torch.save(agent.state_dict(), "cartpole_policy.pth")

        
    
    
if __name__ == "__main__":
    main()




