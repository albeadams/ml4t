import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
    
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_buffer=int(1e6), batch_size=64):
        self.max_size = max_buffer
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_buffer, state_dim))
        self.action = np.zeros((max_buffer, action_dim))
        self.next_state = np.zeros((max_buffer, state_dim))
        self.reward = np.zeros((max_buffer, 1))
        self.not_done = np.zeros((max_buffer, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def update(self, experience):
        state, action, next_state, reward, done = experience
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
# class ReplayBuffer():
#     def __init__(self, 
#                  max_size=10000, 
#                  batch_size=64):
#         self.ss_mem = np.empty(shape=(max_size), dtype=np.ndarray)
#         self.as_mem = np.empty(shape=(max_size), dtype=np.ndarray)
#         self.rs_mem = np.empty(shape=(max_size), dtype=np.ndarray)
#         self.ps_mem = np.empty(shape=(max_size), dtype=np.ndarray)
#         self.ds_mem = np.empty(shape=(max_size), dtype=np.ndarray)

#         self.max_size = max_size
#         self.batch_size = batch_size
#         self._idx = 0
#         self.size = 0
    
#     def store(self, sample):
#         s, a, r, p, d = sample
#         self.ss_mem[self._idx] = s
#         self.as_mem[self._idx] = a
#         self.rs_mem[self._idx] = r
#         self.ps_mem[self._idx] = p
#         self.ds_mem[self._idx] = d
        
#         self._idx += 1
#         self._idx = self._idx % self.max_size

#         self.size += 1
#         self.size = min(self.size, self.max_size)

#     def sample(self, batch_size=None):
#         if batch_size == None:
#             batch_size = self.batch_size

#         idxs = np.random.choice(
#             self.size, batch_size, replace=False)
#         experiences = np.vstack(self.ss_mem[idxs]), \
#                       np.vstack(self.as_mem[idxs]), \
#                       np.vstack(self.rs_mem[idxs]), \
#                       np.vstack(self.ps_mem[idxs]), \
#                       np.vstack(self.ds_mem[idxs])
#         return experiences

#     def __len__(self):
#         return self.size

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# https://github.com/sfujim/TD3/blob/master/TD3.py

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 512)
        self.l4 = nn.Linear(512, 512)
        self.l5 = nn.Linear(512, 256)
        self.l6 = nn.Linear(256, action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = F.relu(self.l4(a))
        a = F.relu(self.l4(a))
        a = F.relu(self.l4(a))
        a = F.relu(self.l5(a))
        return self.max_action * torch.tanh(self.l6(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 512)
        self.l4 = nn.Linear(512, 512)
        self.l5 = nn.Linear(512, 256)
        self.l6 = nn.Linear(256, 1)

        # Q2 architecture
        self.l7 = nn.Linear(state_dim + action_dim, 256)
        self.l8 = nn.Linear(256, 256)
        self.l9 = nn.Linear(256, 512)
        self.l10 = nn.Linear(512, 512)
        self.l11 = nn.Linear(512, 256)
        self.l12 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = F.relu(self.l4(q1))
        q1 = F.relu(self.l4(q1))
        q1 = F.relu(self.l4(q1))
        q1 = F.relu(self.l5(q1))
        q1 = self.l6(q1)

        q2 = F.relu(self.l7(sa))
        q2 = F.relu(self.l8(q2))
        q2 = F.relu(self.l9(q2))
        q2 = F.relu(self.l10(q2))
        q2 = F.relu(self.l10(q2))
        q2 = F.relu(self.l10(q2))
        q2 = F.relu(self.l11(q2))
        q2 = self.l12(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = F.relu(self.l4(q1))
        q1 = F.relu(self.l4(q1))
        q1 = F.relu(self.l4(q1))
        q1 = F.relu(self.l5(q1))
        q1 = self.l6(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state.float()).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample()
        state = torch.tensor(state).float()
        action = torch.tensor(action).float()
        next_state = torch.tensor(next_state).float()
        reward = torch.tensor(reward).float()
        not_done = torch.tensor(not_done).float()
        #print(state, action, next_state, reward, not_done)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)
            
            next_action = torch.tensor(next_action).float()

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        #critic_loss = critic_loss.double()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        
        
        
        
class DDPG(object):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 max_action, 
                 discount=0.99, 
                 tau=0.005,
                 policy_noise=None, # unused
                 noise_clip=None,
                 policy_freq=None):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau


    def select_action(self, state):
#         state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
#         return self.actor(state).cpu().data.numpy().flatten()
        state = torch.tensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state.float()).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=256):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample()
        state = torch.tensor(state).float()
        action = torch.tensor(action).float()
        next_state = torch.tensor(next_state).float()
        reward = torch.tensor(reward).float()
        not_done = torch.tensor(not_done).float()

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        print(next_state)
        print(target_Q)
        print(reward)
        print(not_done)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        
        
        
        
# class FCDuelingQ(nn.Module):
#     def __init__(self, 
#                  input_dim, 
#                  output_dim, 
#                  hidden_dims=(32,32), 
#                  activation_fc=F.relu):
#         super(FCDuelingQ, self).__init__()
#         self.activation_fc = activation_fc

#         self.input_layer = nn.Linear(input_dim, hidden_dims[0])
#         self.hidden_layers = nn.ModuleList()
#         for i in range(len(hidden_dims)-1):
#             hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
#             self.hidden_layers.append(hidden_layer)
#         self.output_value = nn.Linear(hidden_dims[-1], 1)
#         self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

#         device = "cpu"
#         if torch.cuda.is_available():
#             device = "cuda:0"
#         self.device = torch.device(device)
#         self.to(self.device)
        
#     def _format(self, state):
#         x = state
#         if not isinstance(x, torch.Tensor):
#             x = torch.tensor(x, 
#                              device=self.device, 
#                              dtype=torch.float32)
#             x = x.unsqueeze(0)      
#         return x

#     def forward(self, state):
#         x = self._format(state)
#         x = self.activation_fc(self.input_layer(x))
#         for hidden_layer in self.hidden_layers:
#             x = self.activation_fc(hidden_layer(x))
#         a = self.output_layer(x)
#         v = self.output_value(x).expand_as(a)
#         q = v + a - a.mean(1, keepdim=True).expand_as(a)
#         return q

#     def numpy_float_to_device(self, variable):
#         variable = torch.from_numpy(variable).float().to(self.device)
#         return variable

#     def load(self, experiences):
#         states, actions, new_states, rewards, is_terminals = experiences
#         states = torch.from_numpy(states).float().to(self.device)
#         actions = torch.from_numpy(actions).long().to(self.device)
#         new_states = torch.from_numpy(new_states).float().to(self.device)
#         rewards = torch.from_numpy(rewards).float().to(self.device)
#         is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
#         return states, actions, new_states, rewards, is_terminals
        
        
# class EGreedyExpStrategy():
#     def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
#         self.epsilon = init_epsilon
#         self.init_epsilon = init_epsilon
#         self.decay_steps = decay_steps
#         self.min_epsilon = min_epsilon
#         self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
#         self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
#         self.t = 0
#         self.exploratory_action_taken = None

#     def _epsilon_update(self):
#         self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
#         self.t += 1
#         return self.epsilon

#     def select_action(self, model, state):
#         self.exploratory_action_taken = False
#         with torch.no_grad():
#             q_values = model(state).detach().cpu().data.numpy().squeeze()

#         if np.random.rand() > self.epsilon:
#             action = np.argmax(q_values)
#         else:
#             action = np.random.randint(len(q_values))

#         self._epsilon_update()
#         self.exploratory_action_taken = action != np.argmax(q_values)
#         return action
    
# class GreedyStrategy():
#     def __init__(self):
#         self.exploratory_action_taken = False

#     def select_action(self, model, state):
#         with torch.no_grad():
#             q_values = model(state).cpu().detach().data.numpy().squeeze()
#             return np.argmax(q_values)
        
        
# class DuelingDDQN():
#     def __init__(self, state_dim, action_dim, max_action, lr = 0.0005,
#                  value_model_fn, 
#                  value_optimizer_fn, 
#                  value_optimizer_lr,
#                  max_gradient_norm,
#                  training_strategy_fn,
#                  evaluation_strategy_fn,
#                  n_warmup_batches,
#                  update_target_every_steps,
#                  tau):
        
#         self.target_model = FCDuelingQ(state_dim, action_dim, hidden_dims=(512,128))
#         self.online_model = FCDuelingQ(state_dim, action_dim, hidden_dims=(512,128))
#         self.lr = lr
        
#         self.optimizer = optim.RMSprop(self.online_model.parameters(), lr=lr)
        
#         self.max_gradient_norm = float('inf')

#         self.training_strategy = EGreedyExpStrategy(init_epsilon=1.0,  
#                                                     min_epsilon=0.3, 
#                                                     decay_steps=20000)
#         self.evaluation_strategy = GreedyStrategy()
        
        
#     def optimize_model(self, experiences):
#         states, actions, rewards, next_states, is_terminals = experiences
#         batch_size = len(is_terminals)

#         argmax_a_q_sp = self.online_model(next_states).max(1)[1]
#         q_sp = self.target_model(next_states).detach()
#         max_a_q_sp = q_sp[
#             np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)
#         target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
#         q_sa = self.online_model(states).gather(1, actions)

#         td_error = q_sa - target_q_sa
#         value_loss = td_error.pow(2).mul(0.5).mean()
#         self.value_optimizer.zero_grad()
#         value_loss.backward()        
#         torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 
#                                        self.max_gradient_norm)
#         self.value_optimizer.step()
        
        
#     def train(self, buffer, batch_size):
        
#         self.update_network(tau=1.0)
        
        
        