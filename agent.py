import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayBufferNumpy
import numpy as np
import pickle

# GPU availability check
if torch.cuda.is_available():
    print("CUDA (GPU support) is available!")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Running on CPU.")


# Define the Deep Q-Network (DQN) class
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, num_input_channels=2):
        super(DQN, self).__init__()

        
        # Convolutional layers with relu activation
        self.conv = nn.Sequential(
            nn.Conv2d(num_input_channels, 16, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.ReLU()
        )
        
        # Calculate the size of the tensor after the conv layers
        self.in_features = self._calculate_conv_output_size(input_shape, num_input_channels)
        
        
        # Dense layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        # Pass the input 'x' through the convolutional layers
        x = self.conv(x)
        # Reshape the output tensor to prepare it for the fully connected layers
        x = x.reshape(x.size(0), -1)
        # Pass the reshaped tensor through the fully connected layers
        x = self.fc(x)
        
        return x
    
    def predict(self, state):
        # This method takes a state as input and returns the Q values ​​for all actions
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)  # Convert to the correct format and add a batch dimension
            q_values = self.forward(state)
        return q_values.numpy()

    def _calculate_conv_output_size(self, input_shape, num_input_channels):
        test_tensor = torch.zeros(1, num_input_channels, *input_shape)
        test_tensor = self.conv(test_tensor)
        total_size = test_tensor.size(1) * test_tensor.size(2) * test_tensor.size(3)
        return total_size

# Define the DeepQLearningAgent class    
class DeepQLearningAgent(nn.Module):
    def __init__(self, board_size, frames, n_actions, gamma=0.99, buffer_size=1000, use_target_net=True, model_config=None,version=None):
        super(DeepQLearningAgent, self).__init__()
        
        self.board_size = board_size
        self.n_frames = frames
        self.n_actions = n_actions
        self.gamma = gamma
        self.use_target_net = use_target_net
        self.version = version
        self.input_shape = (board_size, board_size)

        self.buffer = ReplayBufferNumpy(buffer_size, board_size, frames, n_actions)

        self.model = self._build_model(version)
        if self.use_target_net:
            self.target_model = self._build_model(version)
            self._update_target_model()

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.0005)

        # Add this line to create self.device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, version):
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
        # Normalize them to a range between 0 and 1
        return board.astype(np.float32) / 4.0

    def reset_buffer(self, buffer_size=None):
        # Reset the replay buffer with a new buffer size if provided
        if buffer_size is not None:
            self._buffer_size = buffer_size
        # Create a new replay buffer with the updated buffer size and other parameters   
        self.buffer = ReplayBufferNumpy(self._buffer_size, self.board_size, self.n_frames, self.n_actions)

    def get_buffer_size(self):
        # Get the current size of the replay buffer
        return len(self.buffer)

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        # Add a new experience tuple to the replay buffer
        self.buffer.add_to_buffer(board, action, reward, next_board, done, legal_moves)

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
        # Initialize the total loss to zero
        total_loss = 0.0
        
        # Define the loss function (Smooth L1 Loss)
        loss_fn = nn.SmoothL1Loss()
        
         # Iterate over a specified number of games for training
        for _ in range(num_games):
            # Sample experiences (states, actions, rewards, next_states, dones, legal_moves) from the replay buffer
            states, actions, rewards, next_states, dones, legal_moves = self.buffer.sample(batch_size)

            # Check if states and next_states need to be squeezed to match shapes
            if states.shape != next_states.shape:
                states = torch.squeeze(states, dim=1)
                next_states = torch.squeeze(next_states, dim=1)

            # Convert numpy arrays to PyTorch tensors and reshape them
            states = torch.from_numpy(states).float().permute(0, 3, 1, 2).to(self.device)
            next_states = torch.from_numpy(next_states).float().permute(0, 3, 1, 2).to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            dones = torch.from_numpy(dones).float().to(self.device)

            # Clip rewards to be either -1, 0, or 1
            if reward_clip:
                rewards = torch.sign(rewards)

            # Convert actions to PyTorch tensors and choose the actions with the maximum Q values
            actions = torch.from_numpy(actions).to(self.device)
            actions = torch.argmax(actions, axis=1).unsqueeze(-1)

            # Compute the Q values of the current states for the chosen actions
            current_Q_values = self.model(states).gather(1, actions).squeeze(-1)

            # Compute the maximum Q values of the next states without gradients
            with torch.no_grad():
                max_next_Q_values = self.model(next_states).max(1)[0]
                
                # Ensure all components are 1D tensors of shape [batch_size(32)]
                max_next_Q_values = max_next_Q_values.squeeze()
                rewards = rewards.squeeze()
                dones = dones.squeeze()
                
                # Compute expected Q values
                expected_Q_values = rewards + (self.gamma * max_next_Q_values * (1 - dones))

            
            
            # Ensure expected_Q_values is a 1D tensor of shape [batch_size(32)]
            expected_Q_values = expected_Q_values.squeeze()

            # Check if the shapes of current_Q_values and expected_Q_values match
            assert current_Q_values.shape == expected_Q_values.shape, f"Shape mismatch: {current_Q_values.shape} vs {expected_Q_values.shape}"

            # Calculate the loss using the Smooth L1 Loss
            loss = loss_fn(current_Q_values, expected_Q_values.detach())

            # Zero the gradients, perform backpropagation, and update the model's weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # If using a target network, update it with the main model's weights
            if self.use_target_net:
                self._update_target_model()

            # Accumulate the loss
            total_loss += loss.item()

        # Calculate the average loss over the specified number of games
        average_loss = total_loss / num_games
        return average_loss


    def save_model(self, file_path='', iteration=None):
        # Define the file name for saving the main model
        model_file = f"{file_path}/model"
        if iteration is not None:
            model_file += f"_{iteration:04d}"
        model_file += ".pth"

        # Save the state dictionary of the main model to the specified file
        torch.save(self.model.state_dict(), model_file)

        # Check if a target network is used and save its state dictionary
        if self.use_target_net:
            target_model_file = f"{file_path}/target_model_{iteration:04d}.pth"
            torch.save(self.target_model.state_dict(), target_model_file)

    def load_model(self, file_path='', iteration=None):
        # Define the file name for loading the main model
        model_file = f"{file_path}/model"
        if iteration is not None:
            model_file += f"_{iteration:04d}"
        model_file += ".pth"

        # Load the state dictionary of the main model from the specified file
        self.model.load_state_dict(torch.load(model_file))
        
        # Check if a target network is used and load its state dictionary
        if self.use_target_net:
            target_model_file = f"{file_path}/target_model_{iteration:04d}.pth"
            self.target_model.load_state_dict(torch.load(target_model_file))

    def print_models(self):
        # Print the architecture of the main model
        print("Mainmodel:")
        print(self.model)
        
        # Check if a target network is used and print its architecture
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
        # Check if weights of main and target models are identical.
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
        masked_q_values = q_values.detach() + (legal_moves - 1) * 1e9  # large negative value for illegal moves

        # Choose the action with the maximum Q value
        action = torch.argmax(masked_q_values, dim=1)
        
        return action.detach().numpy()

    def get_action_proba(self, board, values=None):
        # Prepare inputs for the model
        board_tensor = self._prepare_input(board)

        # Get Q values ​​from the model
        q_values = self.model(board_tensor)

        # Use Softmax to convert Q values ​​to probabilities
        action_probabilities = F.softmax(q_values, dim=1)
        
        return action_probabilities

    def _point_to_row_col(self, point, board_size):
        # Convert point index to row and column coordinates.
        row = point // board_size
        col = point % board_size
        return row, col

    def _row_col_to_point(self, row, col, board_size):
        # Convert row and column coordinates to point index.
        return row * board_size + col

    def copy_weights_from_agent(self, agent_for_copy):
         # Copy weights from another agent to this agent's models.
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


