from gym_torcs import TorcsEnv
from ddpg import *
import numpy as np
from OU import OU
import os
import math
import sys
# from PIL import Image
# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle

parser = argparse.ArgumentParser()

# Experiments specific parameters
parser.add_argument('--is_learning', action='store_true', help='learning flag')
parser.add_argument('--is_vision', action='store_true', help='available img flag')
parser.add_argument('--is_throttle', action='store_true', help='available throttle flag')
parser.add_argument('--is_rendering', action='store_true', help='available visualization flag')

# Training specific parameters
parser.add_argument('--batch_size', type=int, default=32, help='minibatch size')
parser.add_argument('--episode_count', type=int, default=2000, help='number of episodes')
parser.add_argument('--max_steps', type=int, default=100000, help='number of max steps')
parser.add_argument('--lr_pi', type=float, default=1e-4, help='learning rate of actornet')
parser.add_argument('--lr_v', type=float, default=1e-3, help='learning rate of criticnet')

# Model specific parameters
parser.add_argument('--action_dim', type=int, default=3, help='steer/accel/brake')
parser.add_argument('--state_dim', type=int, default=29, help='angle,track,trackPos,speedX,speedY,speedZ,wheelSpinVel,rpm')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--is_noise', choices=[1, 0], type=int, default=1, help='available noise: 1, else: 2')
parser.add_argument('--tag', default='tag', help='personal tag for the model')

args = parser.parse_args()

print('*'*30)
print("Argument initialting....")
print(args)
print('*'*30)

reward = 0
done = False
step = 0

EXPLORE = 100000.
epsilon = 1

OU = OU()
agent = Agent(batch_size=args.batch_size, state_size=args.state_dim, action_size=args.action_dim, gamma=args.gamma, tau=args.tau, lr_pi=args.lr_pi, lr_v=args.lr_v)

checkpoint_dir = './checkpoint/'+args.tag+'/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)

print('Checkpoint dir:', checkpoint_dir)

writer = SummaryWriter(log_dir="./runs")

# Trainig
metrics = {'max_reward':[], 'avg_reward':[], 'actor_loss':[], 'critic_loss':[]}
constant_metrics = {'max_reward_episodes':-1, 'max_reward':-1e5}

# Generate a Torcs environment
env = TorcsEnv(vision=args.is_vision, throttle=args.is_throttle, rendering=args.is_rendering)

print("TORCS Experiment Start.")
for i in range(args.episode_count):
    # print("Episode : " + str(i))

    if np.mod(i, 3) == 0 and args.is_rendering:
        # Sometimes you need to relaunch TORCS because of the memory leak error
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

    total_reward = 0.
    is_fst_loss = True
    loss_actor = 0
    loss_critic = 0
    for j in range(args.max_steps):
        epsilon -= 1.0 / EXPLORE
        action = agent.get_action(state)
        action = action.detach().cpu().numpy()
        # action = action.squeeze(0)

        action_ = np.zeros([1, args.action_dim])
        noise_ = np.zeros([1, args.action_dim])

        # print(f'steer: {action[0][0]}, accel: {action[0][1]}, brake: {action[0][2]}')

        noise_[0][0] = args.is_noise * max(epsilon, 0) * OU.function(action[0][0], 0.0, 0.60, 0.30) # mu, theta, sigma
        noise_[0][1] = args.is_noise * max(epsilon, 0) * OU.function(action[0][1], 0.5, 1.00, 0.10)
        noise_[0][2] = args.is_noise * max(epsilon, 0) * OU.function(action[0][2], -0.1, 1.00, 0.05)

        action_[0][0] = action[0][0] + noise_[0][0]
        action_[0][1] = action[0][1] + noise_[0][1]
        action_[0][2] = action[0][2] + noise_[0][2]

        ob, reward, done, _ = env.step(action_[0])
        next_state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        agent.add_memory(state, action[0], next_state, reward, done)
        state = next_state
        total_reward += reward

        loss_critic_, loss_actor_ = agent.update()

        step += 1
        if done:
            loss_actor = loss_actor / j
            loss_critic = loss_critic / j
            is_fst_loss = True
            break
        else:
            if is_fst_loss:
                loss_critic = loss_critic_
                loss_actor = loss_actor_
                is_fst_loss = False
            else:
                loss_critic += loss_critic_
                loss_actor += loss_actor_
    
    avg_reward = total_reward / j

    writer.add_scalar("total_reward", total_reward.item(), i)
    writer.add_scalar("avg_reward", avg_reward.item(), i)
    writer.add_scalar("loss/critic", loss_critic, i)
    writer.add_scalar("loss/actor", loss_actor, i)

    metrics['max_reward'].append(total_reward.item())
    metrics['avg_reward'].append(avg_reward.item())
    metrics['critic_loss'].append(loss_critic)
    metrics['actor_loss'].append(loss_actor)

    if metrics['max_reward'][-1] > constant_metrics['max_reward']:
        constant_metrics['max_reward'] = metrics['max_reward'][-1]
        constant_metrics['max_reward_episodes'] = i
        agent.model_save(checkpoint_dir)

    # print('*'*30)
    print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    
    print('Episode:',args.tag,":", i)
    for k,v in metrics.items():
        if len(v)>0:
            print(k,v[-1])

    print(constant_metrics)
    print('*'*30)

    with open(checkpoint_dir+'metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)

# writer.close()
env.end()  # This is for shutting down TORCS
print("Finish.")
