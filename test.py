from gym_torcs import TorcsEnv
import numpy as np

#### Generate a Torcs environment
# enable vision input, the action is steering only (1 dim continuous action)
vision = False
reward = 0
done = False
episodes = 1000
max_steps = 10000
total_reward = 0

env = TorcsEnv(vision=vision, throttle=False)

# without vision input, the action is steering and throttle (2 dim continuous action)
# env = TorcsEnv(vision=False, throttle=True)

# ob = env.reset(relaunch=True)  # with torcs relaunch (avoid memory leak bug in torcs)
# ob = env.reset()  # without torcs relaunch

# Generate an agent
from sample_agent import Agent
agent = Agent(1)  # steering only

for episode in range(episodes):
    ob = env.reset()
    total_reward = 0

    for i in range(max_steps):
        action = agent.act(ob, reward, done, vision)

    # single step
        # action = np.array([0.0])
        ob, reward, done, _ = env.step(action)
        # print(ob["focus"])
        # print(ob)

        total_reward += reward

        if done:
            break
    
    print(episode)

# shut down torcs
env.end()