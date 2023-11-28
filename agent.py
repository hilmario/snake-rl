import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.fc_input_size = self._calculate_conv_output_size(input_shape)

        self.fc1 = nn.Linear(in_features=self.fc_input_size, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

    def _calculate_conv_output_size(self, input_shape):
        test_tensor = torch.zeros(1, *input_shape)
        test_tensor = self.conv1(test_tensor)
        test_tensor = self.conv2(test_tensor)
        test_tensor = self.conv3(test_tensor)
        return int(torch.numel(test_tensor) / test_tensor.size(0))

class DeepQLearningAgent(nn.Module):
    def __init__(self, board_size, n_frames, n_actions, gamma=0.99, buffer_size=10000, use_target_net=True, model_config=None):
        super(DeepQLearningAgent, self).__init__()
        
        self.board_size = board_size
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.gamma = gamma
        self.use_target_net = use_target_net
        self.input_shape = (n_frames, board_size, board_size)

        self.buffer = ReplayBuffer(buffer_size, board_size, n_frames, n_actions)

        self.model = self._build_model(model_config)
        if self.use_target_net:
            self.target_model = self._build_model(model_config)
            self._update_target_model()

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.0005)

    def _build_model(self, model_config):
        return DQN(self.input_shape, self.n_actions)

    def _update_target_model(self):
        # Oppdaterer målnettverket med vekter fra hovedmodellen
        if self.use_target_net:
            self.target_model.load_state_dict(self.model.state_dict())


    def _prepare_input(self, board):
        # Reshape input data if needed
        if board.ndim == 3:
            board = np.expand_dims(board, axis=0)
        
        # Normalize the board data
        board = self._normalize_board(board.copy())  # Implement _normalize_board function
        
        # Convert the board data to PyTorch tensor
        board = torch.tensor(board, dtype=torch.float32)
        
        return board

    def _normalize_board(self, board):
        # Normalize the board state before input to the network
        pass

    def reset_buffer(self, buffer_size=None):
        # Reset or initialize the replay buffer
        pass

    def get_buffer_size(self):
        # Get the current size of the buffer
        pass

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        # Add current game step to the replay buffer
        pass

    def save_buffer(self, file_path='', iteration=None):
        # Save the buffer to disk
        pass

    def load_buffer(self, file_path='', iteration=None):
        # Load the buffer from disk
        pass

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        # Implement the training logic for the agent
        pass

    def save_model(self, file_path='', iteration=None):
        # Save the current models to disk
        pass

    def load_model(self, file_path='', iteration=None):
        # Load models from disk
        pass

    def print_models(self):
        # Print the current models using summary method
        pass

    def set_weights_trainable(self):
        # Set selected layers to non-trainable and compile the model
        pass

    def update_target_net(self):
        # Update the weights of the target network
        pass

    def compare_weights(self):
        # Check if the model and target network have the same weights
        pass

    def move(self, board, legal_moves, value=None):
        # Decide the action with maximum Q value
        pass

    def get_action_proba(self, board, values=None):
        # Returns the action probability values using the model
        pass

    def _point_to_row_col(self, point):
        # Convert a point value to row, col value
        pass

    def _row_col_to_point(self, row, col):
        # Convert a (row, col) to point value
        pass

    def copy_weights_from_agent(self, agent_for_copy):
        # Update weights between agents
        pass

    def reset_models(self):
        # Reset or recreate the models
        pass



