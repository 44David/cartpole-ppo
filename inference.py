import gymnasium as gym
from policy import CartPoleAgent
import torch


def main():
    
    agent = CartPoleAgent()
    
    agent.load_state_dict(torch.load("cartpole_policy.pth"))
    agent.eval()
    
    env = gym.make("CartPole-v1", render_mode="human")
    for episode in range(5):
        observation, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent(torch.Tensor(observation)).argmax()
            observation, reward, terminated, truncated, info = env.step(action.item())
            env.render()
            total_reward += 1
            
            done = terminated or truncated
        print(total_reward)
            
            
            
if __name__ == "__main__":
    main()