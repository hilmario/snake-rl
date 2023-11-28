import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from replay_buffer import ReplayBufferNumpy
from collections import deque


def huber_loss(y_true, y_pred, delta=1):
    error = y_true - y_pred
    is_small_error = torch.abs(error) < delta
    squared_loss = torch.square(error) / 2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small_error, squared_loss, linear_loss)

def mean_huber_loss(y_true, y_pred, delta=1):
    return huber_loss(y_true, y_pred, delta).mean()


class Agent():
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2).reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, self._n_frames, self._n_actions)

    def get_buffer_size(self):
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        self._buffer.add_to_buffer(board, action, reward, next_board, done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        return row*self._board_size + col
    
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flatten =nn.Flatten()
        self.fc1 = nn.Linear(64 * input_shape[1] * input_shape[2], 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    


class DeepQLearningAgent(Agent):
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version='', input_channels=2):
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        self.input_channels = input_channels
        self.reset_models()


    def reset_models(self):
        self._model, self._optimizer = self._agent_model()
        if self._use_target_net:
            self._target_net, _ = self._agent_model()
            self.update_target_net()


    def _prepare_input(self, board):
        
        if isinstance(board, np.ndarray):
            board = torch.tensor(board, d=type.float32)

        if board.ndim ==3:
            board = board.unsqueeze(0)

        board = self._normalize_board(board) # normalize
        board = board.permute(0, 3, 1, 2) # change to (batch, channels, height, width)
        expected_channels = 10
        if board.size(1) != expected_channels:
            raise ValueError(f"input 'board' has {board.size(1)} channels, but {expected_channels} channels are expected.")

        return board

    def _get_model_outputs(self, board, model=None):

        board = self._prepare_input(board)  # Prepare board

        if model is None:
            model = self._model

        with torch.no_grad():
            model_outputs = model(board)  # Get model predictions

        return model_outputs.cpu().numpy()
   

    def _normalize_board(self, board):
        return board/4.0


    def move(self, board, legal_moves, value=None):
        model_outputs = self._get_model_outputs(board, self._model)
        legal_moves = torch.tensor(legal_moves).unsqueeze(0) if legal_moves.ndim == 1 else torch.tensor(legal_moves)
        return torch.argmax(torch.where(legal_moves == 1, torch.tensor(model_outputs), torch.tensor(-np.inf)), axis=1).numpy()
        
        


    def _agent_model(self):
        input_channels = 10
        input_height = self._board_size
        input_width = self._board_size
        model = DQN(input_shape=(input_channels, input_height, input_width), n_actions= self._n_actions)
        optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
        
        return model, optimizer
    


    


    def set_weights_trainable(self):
        for param in self._model.parameters():
            param.requires_grad = False
       
        for param in self._model[-1].parameters():
            param.requires_grad = True

    def get_action_proba(self, board, values=None):
        model_outputs = self._get_model_outputs(board, self._model)
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - np.max(model_outputs, axis=1).reshape((-1, 1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs / np.sum(model_outputs, axis=1).reshape((-1, 1))
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        torch.save(self._model.state_dict(), "{}/model_{:04d}.pt".format(file_path, iteration))
        if self._use_target_net:
            torch.save(self._target_net.state_dict(), "{}/model_{:04d}_target.pt".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_state_dict(torch.load("{}/model_{:04d}.pt".format(file_path, iteration)))
        if self._use_target_net:
            self._target_net.load_state_dict(torch.load("{}/model_{:04d}_target.pt".format(file_path, iteration)))

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        self._model.train()  # Set the model to training mode
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)

        if reward_clip:
            r = np.sign(r)

        # Convert to PyTorch tensors
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float32)
        next_s = torch.tensor(next_s, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Get current Q values
        current_q_values = self._model(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Compute the expected Q values
        next_q_values = self._target_net(next_s).max(1)[0]
        expected_q_values = r + self._gamma * next_q_values * (1 - done)

        # Compute Huber loss
        loss = mean_huber_loss(current_q_values, expected_q_values)

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()       
        

    def update_target_net(self):
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def compare_weights(self):
        for model_param, target_param in zip(self.model.parameters(), self._target_net.parameters()):
            if torch.equal(model_param, target_param):
                print("Weights match")
            else:
                print("Weights do not match")
        

    def copy_weights_from_agent(self, agent_for_copy):
        assert isinstance(agent_for_copy, self), "Agent type is required for copy"
        self._model.load_state_dict(agent_for_copy._model.state_dict())
        if self._use_target_net:
            self._target_net.load_state_dict(agent_for_copy._model.state_dict())



