import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

VIDEO_BIT_RATES = [398, 802, 1203, 2406, 4738] # kbits per sec
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
STATE_LEN = 6 # Number of state parameters
STATE_MEMORY_LEN = 8 # Number of frames to consider from the past
ACTION_LEN = 5 # Number of possible discrete actions (bit-rates of the next chunk)
BUFFER_NORM_FACTOR = 10.0*1000.0 # 10 seconds, convert from milliseconds
HIDDEN_LAYER_SIZE = 256
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001

# Define the Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        distribution = Categorical(F.softmax(self.fc2(x)))
        return distribution

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x


# Define the Actor-Critic agent
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=ACTOR_LEARNING_RATE)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=CRITIC_LEARNING_RATE)

    def save(self, episodeNum, folder):
        torch.save(self.actor.state_dict(), './' + folder + '/actor_episode_{}.pth'.format(episodeNum))
        torch.save(self.critic.state_dict(), './' + folder + '/critic_episode_{}.pth'.format(episodeNum))

    def load(self, folder):
        self.actor.load_state_dict(torch.load(folder + '/actor.pth'))
        self.critic.load_state_dict(torch.load(folder + '/critic.pth'))

    def calculate_reward(self, input_reward, state):
        reward = input_reward["bitrate"] / 1000.0 \
                    - REBUF_PENALTY * input_reward["rebuffer"] / 1000.0 \
              - SMOOTH_PENALTY * np.abs(input_reward["bitrate"] -
                                    state["last"]) / 1000.0
        return reward

    def _state_to_inputs(self, input_state):
        state = {}
        state["x0"] = [input_state["last"]] / np.max(VIDEO_BIT_RATES)  # Prev bit rate
        state["x1"] = [input_state["buffer"] / BUFFER_NORM_FACTOR]  # Buffer size

        x2 = input_state["throughput"]
        if (len(x2) < STATE_MEMORY_LEN):
            x2 = np.concatenate((np.zeros(STATE_MEMORY_LEN - len(x2)), x2))
        elif (len(x2) > STATE_MEMORY_LEN):
            x2 = x2[-STATE_MEMORY_LEN:]

        state["x2"] = [[[x / 8000.0 for x in x2]]] # Convert kbps to mega bytes per second

        x3 = input_state["delay"]
        if (len(x3) < STATE_MEMORY_LEN):
            x3 = np.concatenate((np.zeros(STATE_MEMORY_LEN - len(x3)), x3))
        elif (len(x3) > STATE_MEMORY_LEN):
            x3 = x3[-STATE_MEMORY_LEN:]
        state["x3"] = [[[x / BUFFER_NORM_FACTOR for x in x3]]]
        state["x4"] = [input_state["bitrate"]]  # size of next chunks in mega bytes
        state["x5"] = [input_state["remain"]]  # fraction of chunks to go
        return state

    def parse_state(self, input_state):
        state = [input_state["buffer"]/10000.0 , input_state["last"] / np.max(VIDEO_BIT_RATES), input_state["remain"]]

        tputs = input_state["throughput"]
        if (len(tputs) < STATE_MEMORY_LEN):
            tputs = np.concatenate((np.zeros(STATE_MEMORY_LEN - len(tputs)), tputs))
        elif (len(tputs) > STATE_MEMORY_LEN):
            tputs = tputs[-STATE_MEMORY_LEN:]

        for tput in tputs:
            state.append(tput/8000.0)

        delays = input_state["delay"]
        if (len(delays) < STATE_MEMORY_LEN):
            delays = np.concatenate((np.zeros(STATE_MEMORY_LEN - len(delays)), delays))
        elif (len(delays) > STATE_MEMORY_LEN):
            delays = delays[-STATE_MEMORY_LEN:]

        for delay in delays:
            state.append(delay / 10000.0)

        for nextSize in input_state["nextSizes"]:
            state.append(nextSize)

        return state

    def get_action(self, state):
        state = self.parse_state(state)
        state = torch.tensor(state, dtype=torch.float32)
        act_dist = self.actor(state)
        action = act_dist.sample()
        # log_prob = act_dist.log_prob(action).unsqueeze(0)
        # value = self.critic(state)
        return action.cpu().numpy()

    def update(self, state, action, reward, next_state, done, gamma):
        rewardTrain = self.calculate_reward(reward, state)
        state = self.parse_state(state)
        next_state = self.parse_state(next_state)

        # state = self._state_to_inputs(state)
        # next_state = self._state_to_inputs(next_state)
        state = torch.tensor(state, dtype=torch.float32)
        # action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(rewardTrain, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # Compute TD error
        value = self.critic(state)
        next_value = self.critic(next_state)
        td_target = reward + gamma * next_value * (1 - done)
        td_error = td_target - value

        # Update the critic
        critic_loss = td_error.pow(2)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update the actor
        act_dist = self.actor(state)
        action = torch.tensor(act_dist.sample(), dtype=torch.long)
        log_prob = act_dist.log_prob(action).unsqueeze(0)

        # action_probs = self.actor(state)
        # log_probs = torch.log(action_probs)
        # selected_log_prob = log_probs[action]
        actor_loss = -log_prob * td_error.detach()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        return rewardTrain
