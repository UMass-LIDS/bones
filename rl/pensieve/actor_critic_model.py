import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from collections import namedtuple
import torch.nn.functional as F
from torch.distributions import Categorical


STATE_LEN = 6 # Number of state parameters
STATE_MEMORY_LEN = 8 # Number of frames to consider from the past
ACTION_LEN = 5 # Number of possible discrete actions (bit-rates of the next chunk)

ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
RAND_RANGE = 1000000
DISCOUNT_FACTOR = 0.99
TRAIN_SEQ_LEN = 16
M_IN_K = 1000.0
VIDEO_BIT_RATES = [398, 802, 1203, 2406, 4738]
BUFFER_NORM_FACTOR = 10.0 * 1000.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1

class ActorCritic():
    def __init__(self, is_central):
        # create actor
        self.discount = 0.99
        self.entropy_weight = 0.5
        self.entropy_eps = 1e-6
        self._is_central = is_central

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._actor = Actor().to(self.device)

        if self._is_central:
            # create critic
            self._actor_optimizer = optim.RMSprop(self._actor.parameters(), lr=ACTOR_LR_RATE, alpha=0.9, eps=1e-10)
            self._actor_optimizer.zero_grad()
            self._critic = Critic().to(self.device)
            self.criticOptim = torch.optim.RMSprop(self._critic.parameters(), lr=CRITIC_LR_RATE, alpha=0.9, eps=1e-10)
            self._critic_optimizer = optim.RMSprop(self._critic.parameters(), CRITIC_LR_RATE)
            self._critic_optimizer.zero_grad()
        else:
            self._actor.eval()

        self.loss_function=nn.MSELoss()

        # self._log_probs = []
        # self._import env
        # pred_rewards = []
        # self._a_batch = []
        # self._r_batch = []
        # self._s_batch = []
        # self._prev_action = 0
        # self._state = [np.zeros((STATE_LEN, ACTION_LEN))]
        # self._critic_loss = torch.nn.MSELoss()

    def getNetworkGradient(self, s_batch, a_batch, r_batch, terminal):
        s_batch = torch.cat(s_batch).to(self.device)
        a_batch = torch.LongTensor(a_batch).to(self.device)
        r_batch = torch.tensor(r_batch).to(self.device)
        R_batch = torch.zeros(r_batch.shape).to(self.device)

        R_batch[-1] = r_batch[-1]
        for t in reversed(range(r_batch.shape[0] - 1)):
            R_batch[t] = r_batch[t] + self.discount * R_batch[t + 1]

        with torch.no_grad():
            v_batch = self._critic.forward(s_batch).squeeze().to(self.device)
        td_batch = R_batch - v_batch

        probability = self._actor.forward(s_batch)
        m_probs = Categorical(probability)
        log_probs = m_probs.log_prob(a_batch)
        actor_loss = torch.sum(log_probs * (-td_batch))
        entropy_loss = -self.entropy_weight * torch.sum(m_probs.entropy())
        actor_loss = actor_loss + entropy_loss
        actor_loss.backward()

        # original
        critic_loss = self.loss_function(R_batch, self._critic.forward(s_batch).squeeze())

        critic_loss.backward()

    def actionSelect(self, stateInputs):
        if not self._is_central:
            with torch.no_grad():
                probability = self._actor.forward(stateInputs)
                m = Categorical(probability)
                action = m.sample().item()
                return action

    def hardUpdateActorNetwork(self, actor_net_params):
        for target_param, source_param in zip(self._actor.parameters(), actor_net_params):
            target_param.data.copy_(source_param.data)

    def updateNetwork(self):
        if self._is_central:
            # use the feature of accumulating gradient in pytorch
            self._actor_optimizer.step()
            self._actor_optimizer.zero_grad()
            self._critic_optimizer.step()
            self._critic_optimizer.zero_grad()

    def getActorParam(self):
        return list(self._actor.parameters())

    def getCriticParam(self):
        return list(self._critic.parameters())

    # def act(self, state):
    #     state = self._state_to_inputs(state)
    #     actions = self._actor(state)
    #     actionsDist = torch.distributions.Categorical(actions)
    #     action = actionsDist.sample()
    #     actionLogProb = actionsDist.log_prob(action)
    #     self._log_probs.append(actionLogProb)
    #     return action.item()
    #
    # def save(self, episodeNum):
    #     torch.save(self._actor.state_dict(), './actor_episode_{}.pth'.format(episodeNum))
    #     torch.save(self._actor.state_dict(), './critic_episode_{}.pth'.format(episodeNum))
    #
    # def load(self):
    #     self._actor.load_state_dict('./actor.pth')
    #     self._critic.load_state_dict('./critic.pth')
    #
    # def _state_to_inputs(self, input_state):
    #     state = {}
    #     state["x0"] = [input_state["last"]] / np.max(VIDEO_BIT_RATES)  # Prev bit rate
    #     state["x1"] = [input_state["buffer"] / BUFFER_NORM_FACTOR]  # Buffer size
    #
    #     x2 = input_state["throughput"]
    #     if (len(x2) < STATE_MEMORY_LEN):
    #         x2 = np.concatenate((np.zeros(STATE_MEMORY_LEN - len(x2)), x2))
    #     elif (len(x2) > STATE_MEMORY_LEN):
    #         x2 = x2[-STATE_MEMORY_LEN:]
    #
    #     state["x2"] = [[[x / 8000.0 for x in x2]]] # Convert kbps to mega bytes per second
    #
    #     x3 = input_state["delay"]
    #     if (len(x3) < STATE_MEMORY_LEN):
    #         x3 = np.concatenate((np.zeros(STATE_MEMORY_LEN - len(x3)), x3))
    #     elif (len(x3) > STATE_MEMORY_LEN):
    #         x3 = x3[-STATE_MEMORY_LEN:]
    #     state["x3"] = [[[x / BUFFER_NORM_FACTOR for x in x3]]]
    #     state["x4"] = [input_state["bitrate"]]  # size of next chunks in mega bytes
    #     state["x5"] = [input_state["remain"]]  # number of chunks to go
    #
    #     return state
    #
    # def train(self, state, action, reward, end_of_video):
    #     # self._updateState(state)
    #     self._s_batch.append(self._state_to_inputs(state))
    #     self._a_batch.append(action)
    #
    #     # reward is video quality - rebuffer penalty - smooth penalty
    #     reward = reward["bitrate"] / M_IN_K \
    #              - REBUF_PENALTY * reward["rebuffer"] \
    #              - SMOOTH_PENALTY * np.abs(reward["bitrate"] -
    #                                        state["last"]) / M_IN_K
    #
    #     self._r_batch.append(reward)
    #
    #     if len(self._a_batch) > TRAIN_SEQ_LEN or end_of_video:
    #         self._train()
    #
    # def updateState(self, newState):
    #     self._state = np.roll(self._state, -1, axis=1)
    #     # this should be S_INFO number of terms
    #     self._state[0, -1] = VIDEO_BIT_RATES[self._prev_action] / float(np.max(VIDEO_BIT_RATES))  # last quality
    #     self._state[1, -1] = newState["buffer"] / BUFFER_NORM_FACTOR  # 10 sec
    #     self._state[2, -1] = float(newState["video_chunk_size"]) / float(newState["delay"]) / M_IN_K  # kilo byte / ms
    #     self._state[3, -1] = float(newState["delay"]) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
    #     self._state[4, :ACTION_LEN] = np.array(newState["bitrate"]) / M_IN_K / M_IN_K  # mega byte
    #     self._state[5, -1] = np.minimum(newState["remain"], CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
    #
    # def _train(self):
    #     observed_rewards = []
    #     observed_reward = 0
    #     pred_rewards = []
    #     for (reward, state) in zip(self._r_batch[::-1], self._s_batch):
    #         observed_reward = reward + DISCOUNT_FACTOR*observed_reward
    #         observed_rewards.insert(0, observed_reward)
    #         pred_rewards.append(self._critic(state))
    #         # loss +=
    #
    #     observed_rewards = torch.tensor(observed_rewards).float()
    #     pred_rewards = torch.tensor(pred_rewards, requires_grad=True).float()
    #     self._critic_optimizer.zero_grad()
    #     loss = F.mse_loss(pred_rewards, observed_rewards)
    #     loss.backward()
    #     self._critic_optimizer.step()
    #
    #     actor_pred = []
    #     actor_observed = []
    #     # iterate over observed rewards
    #     for (log_prob, pred_reward, observed_reward) in zip(self._log_probs, pred_rewards, observed_rewards):
    #         # Calculate difference from predicted rewards.
    #         actor_pred.append(-log_prob*pred_reward)
    #         actor_observed.append(-log_prob*observed_reward)
    #     actor_loss = F.l1_loss(torch.tensor(actor_pred, requires_grad=True).float(), torch.tensor(actor_observed).float())
    #     actor_loss.backward()
    #
    #     # Reset gradients
    #     self._actor_optimizer.zero_grad()
    #     # RMS optimization to apply gradients
    #     self._actor_optimizer.step()
    #
    #     del self._log_probs[:]
    #     del self._s_batch[:]
    #     del self._a_batch[:]
    #     del self._r_batch[:]


class Actor(nn.Module):
    def __init__(self, n_conv=128, n_fc=128, n_fc1=128):
        # create actor model
        super(Actor, self).__init__()

        self.vectorOutDim = n_conv
        self.scalarOutDim = n_fc
        self.numFcInput = 2 * self.vectorOutDim * (STATE_MEMORY_LEN - 4 + 1) + 3 * self.scalarOutDim + self.vectorOutDim * (ACTION_LEN- 4 + 1)
        self.numFcOutput = n_fc1

        self.tConv1d = nn.Conv1d(1, self.vectorOutDim, 4)
        self.dConv1d = nn.Conv1d(1, self.vectorOutDim, 4)
        self.cConv1d = nn.Conv1d(1, self.vectorOutDim, 4)
        self.bufferFc = nn.Linear(1, self.scalarOutDim)
        self.leftChunkFc = nn.Linear(1, self.scalarOutDim)
        self.bitrateFc = nn.Linear(1, self.scalarOutDim)
        self.fullyConnected = nn.Linear(self.numFcInput, self.numFcOutput)
        self.outputLayer = nn.Linear(self.numFcOutput, ACTION_LEN)

        # ------------------init layer weight--------------------
        # tensorflow-1.12 uses glorot_uniform(also called xavier_uniform) to initialize weight
        # uses zero to initialize bias
        # Conv1d also use same initialize method
        nn.init.xavier_uniform_(self.bufferFc.weight.data)
        nn.init.constant_(self.bufferFc.bias.data, 0.0)
        nn.init.xavier_uniform_(self.leftChunkFc.weight.data)
        nn.init.constant_(self.leftChunkFc.bias.data, 0.0)
        nn.init.xavier_uniform_(self.bitrateFc.weight.data)
        nn.init.constant_(self.bitrateFc.bias.data, 0.0)
        nn.init.xavier_uniform_(self.fullyConnected.weight.data)
        nn.init.constant_(self.fullyConnected.bias.data, 0.0)
        nn.init.xavier_uniform_(self.tConv1d.weight.data)
        nn.init.constant_(self.tConv1d.bias.data, 0.0)
        nn.init.xavier_uniform_(self.dConv1d.weight.data)
        nn.init.constant_(self.dConv1d.bias.data, 0.0)
        nn.init.xavier_normal_(self.cConv1d.weight.data)
        nn.init.constant_(self.cConv1d.bias.data, 0.0)


    def forward(self, inputs):
        bitrateFcOut = F.relu(self.bitrateFc(inputs[:, 0:1, -1]), inplace=True)
        bufferFcOut = F.relu(self.bufferFc(inputs[:, 1:2, -1]), inplace=True)
        tConv1dOut = F.relu(self.tConv1d(inputs[:, 2:3, :]), inplace=True)
        dConv1dOut = F.relu(self.dConv1d(inputs[:, 3:4, :]), inplace=True)
        cConv1dOut = F.relu(self.cConv1d(inputs[:, 4:5, :ACTION_LEN]), inplace=True)
        leftChunkFcOut = F.relu(self.leftChunkFc(inputs[:, 5:6, -1]), inplace=True)
        t_flatten = tConv1dOut.view(tConv1dOut.shape[0], -1)
        d_flatten = dConv1dOut.view(dConv1dOut.shape[0], -1)
        c_flatten = cConv1dOut.view(dConv1dOut.shape[0], -1)
        fullyConnectedInput = torch.cat([bitrateFcOut, bufferFcOut, t_flatten, d_flatten, c_flatten, leftChunkFcOut], 1)
        fcOutput = F.relu(self.fullyConnected(fullyConnectedInput), inplace=True)
        out = torch.softmax(self.outputLayer(fcOutput), dim=-1)
        return out


class Critic(nn.Module):
    def __init__(self, n_conv=128, n_fc=128, n_fc1=128):
        super(Critic, self).__init__()
        # create critic model
        self.vectorOutDim = n_conv
        self.scalarOutDim = n_fc
        self.numFcInput = 2 * self.vectorOutDim * (
                    STATE_MEMORY_LEN - 4 + 1) + 3 * self.scalarOutDim + self.vectorOutDim * (ACTION_LEN - 4 + 1)
        self.numFcOutput = n_fc1

        # ----------define layer----------------------
        self.tConv1d = nn.Conv1d(1, self.vectorOutDim, 4)
        self.dConv1d = nn.Conv1d(1, self.vectorOutDim, 4)
        self.cConv1d = nn.Conv1d(1, self.vectorOutDim, 4)
        self.bufferFc = nn.Linear(1, self.scalarOutDim)
        self.leftChunkFc = nn.Linear(1, self.scalarOutDim)
        self.bitrateFc = nn.Linear(1, self.scalarOutDim)
        self.fullyConnected = nn.Linear(self.numFcInput, self.numFcOutput)
        self.outputLayer = nn.Linear(self.numFcOutput, 1)

        # ------------------init layer weight--------------------
        # tensorflow-1.12 uses glorot_uniform(also called xavier_uniform) to initialize weight
        # uses zero to initialize bias
        # Conv1d also use same initialize method
        nn.init.xavier_uniform_(self.bufferFc.weight.data)
        nn.init.constant_(self.bufferFc.bias.data, 0.0)
        nn.init.xavier_uniform_(self.leftChunkFc.weight.data)
        nn.init.constant_(self.leftChunkFc.bias.data, 0.0)
        nn.init.xavier_uniform_(self.bitrateFc.weight.data)
        nn.init.constant_(self.bitrateFc.bias.data, 0.0)
        nn.init.xavier_uniform_(self.fullyConnected.weight.data)
        nn.init.constant_(self.fullyConnected.bias.data, 0.0)
        nn.init.xavier_uniform_(self.tConv1d.weight.data)
        nn.init.constant_(self.tConv1d.bias.data, 0.0)
        nn.init.xavier_uniform_(self.dConv1d.weight.data)
        nn.init.constant_(self.dConv1d.bias.data, 0.0)
        nn.init.xavier_normal_(self.cConv1d.weight.data)
        nn.init.constant_(self.cConv1d.bias.data, 0.0)



    def forward(self, inputs):
        bitrateFcOut = F.relu(self.bitrateFc(inputs[:, 0:1, -1]), inplace=True)
        bufferFcOut = F.relu(self.bufferFc(inputs[:, 1:2, -1]), inplace=True)
        tConv1dOut = F.relu(self.tConv1d(inputs[:, 2:3, :]), inplace=True)
        dConv1dOut = F.relu(self.dConv1d(inputs[:, 3:4, :]), inplace=True)
        cConv1dOut = F.relu(self.cConv1d(inputs[:, 4:5, :ACTION_LEN]), inplace=True)
        leftChunkFcOut = F.relu(self.leftChunkFc(inputs[:, 5:6, -1]), inplace=True)
        t_flatten = tConv1dOut.view(tConv1dOut.shape[0], -1)
        d_flatten = dConv1dOut.view(dConv1dOut.shape[0], -1)
        c_flatten = cConv1dOut.view(dConv1dOut.shape[0], -1)
        fullyConnectedInput = torch.cat([bitrateFcOut, bufferFcOut, t_flatten, d_flatten, c_flatten, leftChunkFcOut], 1)
        fcOutput = F.relu(self.fullyConnected(fullyConnectedInput), inplace=True)
        out = self.outputLayer(fcOutput)
        return out

