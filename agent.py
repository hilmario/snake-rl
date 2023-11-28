import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
import numpy as np
import pickle

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
        # Updates the target network with weights from the main model
        if self.use_target_net:
            self.target_model.load_state_dict(self.model.state_dict())


    def _prepare_input(self, board):
        # Check if board is a single game state or a batch of game states
        if board.ndim == 3:
            # Add a batch dimension if it is a single game state 
            board = np.expand_dims(board, axis=0)

        # Normalize the board
        board = self._normalize_board(board)

        # Convert numpy array to PyTorch tensor
        board_tensor = torch.tensor(board, dtype=torch.float32)

        # Reformat tensor if necessary (eg change dimension order)
        # PyTorch expects (batch, channels, height, width)
        board_tensor = board_tensor.permute(0, 3, 1, 2)

        return board_tensor


    def _normalize_board(self, board):
        # Assume that the values ​​in board are between 0 and a maximum number (e.g. 4)
        # Normaliser dem til et område mellom 0 og 1
        board_normalized = board / 4.0

        return board_normalized

    def reset_buffer(self, buffer_size=None):
        if buffer_size is not None:
            self._buffer_size = buffer_size
        self.buffer = ReplayBuffer(self._buffer_size, self.board_size, self.n_frames, self.n_actions)

    def get_buffer_size(self):
        return len(self.buffer)

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        self.buffer.add(board, action, reward, next_board, done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        # Determine file name based on file path and iteration
        file_name = f"{file_path}/buffer"
        if iteration is not None:
            file_name += f"_{iteration:04d}"
        file_name += ".pkl"

        # Save the buffer to disk using pickle
        with open(file_name, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load_buffer(self, file_path='', iteration=None):
        # Determine file name based on file path and iteration
        file_name = f"{file_path}/buffer"
        if iteration is not None:
            file_name += f"_{iteration:04d}"
        file_name += ".pkl"

        # Load the buffer from disk using pickle
        with open(file_name, 'rb') as file:
            self.buffer = pickle.load(file)

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        for _ in range(num_games):
            # Sample experiences from the buffer
            states, actions, rewards, next_states, dones, legal_moves = self.buffer.sample(batch_size)

            # If reward_clipping is enabled, clip the rewards
            if reward_clip:
                rewards = np.clip(rewards, -1, 1)

            # Calculate expected Q values ​​for next states
            # Use the target network for stability if available
            target_network = self.target_model if self.use_target_net else self.model
            next_q_values = target_network.predict(next_states)

            # Calculate target Q values
            max_next_q_values = np.max(next_q_values, axis=1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

            # Update the model using calculated Q values
            self.optimizer.zero_grad()
            current_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            loss = F.mse_loss(current_q_values, target_q_values)
            loss.backward()
            self.optimizer.step()

            # Periodically refresh the target network
            if self.use_target_net:
                self._update_target_model()
        

    def save_model(self, file_path='', iteration=None):
        model_file = f"{file_path}/model"
        if iteration is not None:
            model_file += f"_{iteration:04d}"
        model_file += ".pth"

        torch.save(self.model.state_dict(), model_file)
        if self.use_target_net:
            target_model_file = f"{file_path}/target_model_{iteration:04d}.pth"
            torch.save(self.target_model.state_dict(), target_model_file)

    def load_model(self, file_path='', iteration=None):
        model_file = f"{file_path}/model"
        if iteration is not None:
            model_file += f"_{iteration:04d}"
        model_file += ".pth"

        self.model.load_state_dict(torch.load(model_file))
        if self.use_target_net:
            target_model_file = f"{file_path}/target_model_{iteration:04d}.pth"
            self.target_model.load_state_dict(torch.load(target_model_file))

    def print_models(self):
        print("Mainmodel:")
        print(self.model)
        if self.use_target_net:
            print("\nTargetnetwork:")
            print(self.target_model)

    def set_weights_trainable(self):
        # Set all teams except the last two to non-trainable
        for layer in list(self.model.children())[:-2]:
            for param in layer.parameters():
                param.requires_grad = False

    def update_target_net(self):
        # Copy weights from the main model to the target network
        self.target_model.load_state_dict(self.model.state_dict())

    def compare_weights(self):
        for param_main, param_target in zip(self.model.parameters(), self.target_model.parameters()):
            if param_main.data.ne(param_target.data).sum() > 0:
                return False
        return True

    def move(self, board, legal_moves, value=None):
        # Prepare inputs for the model
        board_tensor = self._prepare_input(board)
        
        # Get Q values ​​from the model
        q_values = self.model(board_tensor)

        # Masks illegal moves
        masked_q_values = q_values + (legal_moves - 1) * 1e9  # large negative value for illegal moves

        # Choose the action with the maximum Q value
        action = torch.argmax(masked_q_values, dim=1)
        
        return action

    def get_action_proba(self, board, values=None):
        # Prepare inputs for the model
        board_tensor = self._prepare_input(board)

        # Get Q values ​​from the model
        q_values = self.model(board_tensor)

        # Use Softmax to convert Q values ​​to probabilities
        action_probabilities = F.softmax(q_values, dim=1)
        
        return action_probabilities

    def _point_to_row_col(self, point, board_size):
        row = point // board_size
        col = point % board_size
        return row, col

    def _row_col_to_point(self, row, col, board_size):
        return row * board_size + col

    def copy_weights_from_agent(self, agent_for_copy):
        self.model.load_state_dict(agent_for_copy.model.state_dict())
        if self.use_target_net:
            self.target_model.load_state_dict(agent_for_copy.target_model.state_dict())

    def reset_models(self):
        # Restore the main model
        self.model = self._build_model()

        # Restore the target network if used
        if self.use_target_net:
            self.target_model = self._build_model()
            self._update_target_net()


