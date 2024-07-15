from gym_torcs import TorcsEnv
from ddpg import *
import numpy as np
from OU import OU
import math
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

is_vision = False
is_throttle = True
is_rendering = True
is_noise = 1
episode_count = 2000
max_steps = 100000
reward = 0
done = False
step = 0

EXPLORE = 100000.
epsilon = 1
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.001
LR_PI = 1e-4
LR_V = 1e-3

ACTION_DIM = 3 # stter/accel/brake
STATE_DIM = 29

OU = OU()
agent = Agent(batch_size=BATCH_SIZE, state_size=STATE_DIM, action_size=ACTION_DIM, gamma=GAMMA, tau=TAU, lr_pi=LR_PI, lr_v=LR_V, path="./models/")
agent.model_load()

# Generate a Torcs environment
env = TorcsEnv(vision=is_vision, throttle=is_throttle, rendering=is_rendering)
ob = env.reset()

print("TORCS Experiment Start.")
while not done:
    # print("Episode : " + str(i))

    # if np.mod(i, 3) == 0 and is_rendering:
    #     # Sometimes you need to relaunch TORCS because of the memory leak error
    #     ob = env.reset(relaunch=True)
    # else:
    state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

    # total_reward = 0.
    # for j in range(max_steps):
    #     loss = 0
    #     epsilon -= 1.0 / EXPLORE
    action = agent.get_action(state)
    action = action.detach().cpu().numpy()
        # action = action.squeeze(0)

        # action_ = np.zeros([1, ACTION_DIM])
        # noise_ = np.zeros([1, ACTION_DIM])

        # print(f'steer: {action[0][0]}, accel: {action[0][1]}, brake: {action[0][2]}')

        # noise_[0][0] = is_noise * max(epsilon, 0) * OU.function(action[0][0], 0.0, 0.60, 0.30)
        # noise_[0][1] = is_noise * max(epsilon, 0) * OU.function(action[0][1], 0.5, 1.00, 0.10)
        # noise_[0][2] = is_noise * max(epsilon, 0) * OU.function(action[0][2], -0.1, 1.00, 0.05)

        # action_[0][0] = action[0][0] + noise_[0][0]
        # action_[0][1] = action[0][1] + noise_[0][1]
        # action_[0][2] = action[0][2] + noise_[0][2]

        # if i < 200:
        #     action_[0][0] = random.uniform(-math.pi, math.pi)


    ob, reward, done, _ = env.step(action[0])
        # next_state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        # agent.add_memory(state, action[0], next_state, reward, done)
        # state = next_state
        # total_reward += reward

        # agent.update()

        # step += 1
        # if done:
        #     break
    
    # writer.add_scalar("reward", total_reward.item(), i)

    # if np.mod(i, 20) == 0:
        # agent.model_save()

    # print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    # print("Total Step: " + str(step))
    # print(f'steer: {action[0][0]}, accel: {action[0][1]}, brake: {action[0][2]}')
    # print("")

writer.close()
env.end()  # This is for shutting down TORCS
print("Finish.")
