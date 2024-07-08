import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=1, padding=0)
        self.act_fc1 = nn.Linear(4 * self.board_width * self.board_height, self.board_width * self.board_height)
        self.val_conv1 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, padding=0)
        self.val_fc1 = nn.Linear(2 * self.board_width * self.board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, inputs):
        # common layers
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_height * self.board_width)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_height * self.board_width)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_act, x_val

class PolicyValueNet():
    """policy-value network """

    def __init__(self, board_width, board_height, model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty

        self.policy_value_net = Net(self.board_width, self.board_height)
        if self.use_gpu:
            self.policy_value_net = self.policy_value_net.cuda()

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=0.02, weight_decay=self.l2_const)

        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file))

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        if self.use_gpu:
            state_batch = state_batch.cuda()
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.cpu().detach().numpy())
        return act_probs, value.cpu().detach().numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_width, self.board_height)).astype("float32")

        current_state = torch.tensor(current_state, dtype=torch.float32)
        if self.use_gpu:
            current_state = current_state.cuda()
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = np.exp(log_act_probs.cpu().detach().numpy().flatten())

        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value.cpu().detach().numpy()

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        """perform a training step"""
        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        mcts_probs = torch.tensor(mcts_probs, dtype=torch.float32)
        winner_batch = torch.tensor(winner_batch, dtype=torch.float32)

        if self.use_gpu:
            state_batch = state_batch.cuda()
            mcts_probs = mcts_probs.cuda()
            winner_batch = winner_batch.cuda()

        self.optimizer.zero_grad()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        log_act_probs, value = self.policy_value_net(state_batch)
        value = value.view(-1)
        value_loss = F.mse_loss(value, winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1))
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)
