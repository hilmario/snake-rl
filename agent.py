import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from replay_buffer import ReplayBufferNumpy
from collections import deque


def huber_loss(y_true, y_pred, delta=1):
    error = y_true - y_pred
    quad_error = 0.5 * error.pow(2)
    lin_error = delta * (error.abs() - 0.5 * delta)
    return torch.where(error.abs() < delta, quad_error, lin_error)

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
        print(f"Original board shape: {board.shape}")
        if board.ndim == 3:
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy())
        print(f"Prepared board shape: {board.shape}")
        return board.copy()

    def _get_model_outputs(self, board, model=None):
        print(f"Input to model: {board.shape}")
        board = self._prepare_input(board)
        if model is None:
            model = self._model
        model_outputs = model(torch.tensor(board, dtype=torch.float32))
        print(f"Model output: {model_outputs.shape}")
        return model_outputs.detach().numpy()

    def _normalize_board(self, board):
        return board.astype(np.float32) / 4.0

    def move(self, board, legal_moves, value=None):
        print(f"Board shape before transpose: {board.shape}")
        board = board.transpose((0, 3, 1, 2))
        print(f"Board shape after transpose: {board.shape}")
        model_outputs = self._get_model_outputs(board, self._model)
        
       
        legal_moves = legal_moves.reshape((legal_moves.shape[0], -1))
    
        return np.argmax(np.where(legal_moves == 1, model_outputs, -np.inf), axis=1)


    def _agent_model(self, input_channels=2):  
  
        input_dim = self._board_size
        hidden_units = 64

        # Dummy input for testing
        sample_input = torch.rand(1, input_channels, input_dim, input_dim)

        model = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, self._n_actions)
        )

       

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
        
        pass

    def update_target_net(self):
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def compare_weights(self):
        
        pass

    def copy_weights_from_agent(self, agent_for_copy):
        assert isinstance(agent_for_copy, self), "Agent type is required for copy"
        self._model.load_state_dict(agent_for_copy._model.state_dict())
        if self._use_target_net:
            self._target_net.load_state_dict(agent_for_copy._model.state_dict())



