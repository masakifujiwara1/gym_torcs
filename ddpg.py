import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torchrl.data import ReplayBuffer, ListStorage
from ReplayBuffer import *
from torch.utils.tensorboard import SummaryWriter

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNet(nn.Module):
    def __init__(self, action_size=1, state_size=3):
        super().__init__()
        self.l1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.l2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.output_steer = nn.Linear(HIDDEN2_UNITS, 1)
        self.output_accel = nn.Linear(HIDDEN2_UNITS, 1)
        self.output_brake = nn.Linear(HIDDEN2_UNITS, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.bn1 = nn.BatchNorm1d(HIDDEN1_UNITS)
        self.bn2 = nn.BatchNorm1d(HIDDEN2_UNITS)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.l1.weight)
        nn.init.kaiming_normal_(self.l2.weight)
        nn.init.xavier_uniform_(self.output_steer.weight)
        nn.init.xavier_uniform_(self.output_accel.weight)
        nn.init.xavier_uniform_(self.output_brake.weight)
        nn.init.zeros_(self.l1.bias)
        nn.init.zeros_(self.l2.bias)
        nn.init.zeros_(self.output_steer.bias)
        nn.init.zeros_(self.output_accel.bias)
        nn.init.zeros_(self.output_brake.bias)

    def forward(self, x):
        # x1 = self.relu(self.l1(x))
        # x2 = self.relu(self.l2(x1))
        x1 = self.relu(self.bn1(self.l1(x)))
        x2 = self.relu(self.bn2(self.l2(x1)))
        output_steer = self.tanh(self.output_steer(x2))
        output_accel = self.sigmoid(self.output_accel(x2))
        output_brake = self.sigmoid(self.output_brake(x2))
        # print(output_steer, torch.cat((output_steer, output_accel, output_brake), dim=1))
        return torch.cat((output_steer, output_accel, output_brake), dim=1)

class CriticNet(nn.Module):
    def __init__(self, action_size=1, state_size=3):
        super().__init__()
        self.s1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.a1 = nn.Linear(action_size, HIDDEN2_UNITS)
        self.s2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.l1 = nn.Linear(HIDDEN2_UNITS, HIDDEN2_UNITS)
        self.l2 = nn.Linear(HIDDEN2_UNITS, action_size)

        self.relu = nn.ReLU()

        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.s1.weight)
        nn.init.xavier_uniform_(self.a1.weight)
        nn.init.xavier_uniform_(self.s2.weight)
        nn.init.kaiming_normal_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.zeros_(self.s1.bias)
        nn.init.zeros_(self.a1.bias)
        nn.init.zeros_(self.s2.bias)
        nn.init.zeros_(self.l1.bias)
        nn.init.zeros_(self.l2.bias)

    def forward(self, x, action):
        s1 = self.relu(self.s1(x))
        a1 = self.a1(action)
        s2 = self.s2(s1)
        x1 = s2 + a1 # sum
        x2 = self.relu(self.l1(x1))
        x3 = self.l2(x2)
        return x3

class Agent:
    def __init__(self, batch_size=64, state_size=29, action_size=1, gamma=0.99, tau=0.2, lr_pi=1e-4, lr_v=1e-3, path="./models/"):
        torch.manual_seed(0)
        np.random.seed(1337)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.gamma = gamma
        self.lr_pi = lr_pi
        self.lr_v = lr_v
        self.action_size = action_size
        self.state_size = state_size
        self.tau = tau

        self.memory = []
        self.actor = ActorNet(action_size=self.action_size, state_size=self.state_size).to(self.device)
        self.critic = CriticNet(action_size=self.action_size, state_size=self.state_size).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), self.lr_pi)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), self.lr_v)
        self.mse = nn.MSELoss()
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer()

        self.path = path
        # self.writer = SummaryWriter(log_dir="./runs")

    def get_action(self, state):
        self.actor.eval()
        state = state[np.newaxis, :]
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        action = self.actor(state_tensor)
        self.actor.train()
        return action

    def add_memory(self, *args):
        self.replay_buffer.append(*args)

    def reset_memory(self):
        self.replay_buffer.reset()

    def sync_net(self):
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
    
    def soft_update(self, target_net, source_net, tau):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()

        state_batch = torch.tensor(batch.state, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(batch.next_state, device=self.device, dtype=torch.float)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float).unsqueeze(1)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float).unsqueeze(1)

        qvalue = self.critic(state_batch, action_batch)
        next_qvalue = self.critic_target(next_state_batch, self.actor_target(next_state_batch))
        target_qvalue = reward_batch + (1 - done_batch) * self.gamma * next_qvalue

        # criticの損失
        loss_critic = self.mse(qvalue, target_qvalue)

        # actorの損失
        loss_actor = -self.critic(state_batch, self.actor(state_batch)).mean()
        
        loss_critic.backward()
        loss_actor.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

        self.soft_update(self.critic_target, self.critic, self.tau)
        self.soft_update(self.actor_target, self.actor, self.tau)

    def model_save(self):
        torch.save(self.actor.state_dict(), self.path + "model_actor_ddpg.pth")
        torch.save(self.critic.state_dict(), self.path + "model_critic_ddpg.pth")

    def model_load(self):
        self.actor.load_state_dict(torch.load(self.path + "model_actor_ddpg.pth"))
        self.critic.load_state_dict(torch.load(self.path + "model_critic_ddpg.pth"))