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



class PolicyGradientAgent(DeepQLearningAgent):
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=False,
                 version=''):
        super().__init__(board_size=board_size, frames=frames,
                         buffer_size=buffer_size, gamma=gamma,
                         n_actions=n_actions, use_target_net=False,
                         version=version)
        self.actor_optimizer = optim.Adam(self._model.parameters(), lr=1e-6)

    def _agent_model(self):
        input_channels = self._n_frames
        input_dim = self._board_size
        hidden_units = 64

        model = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (input_dim - 8) * (input_dim - 8), hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, self._n_actions)
        )

        

        return model



    def train_agent(self, batch_size=32, beta=0.1, normalize_rewards=False,
                    num_games=1, reward_clip=False):
        s, a, r, _, _, _ = self._buffer.sample(self._buffer.get_current_size())
        if normalize_rewards:
            r = (r - np.mean(r)) / (np.std(r) + 1e-8)
        
        
        # Convert to PyTorch tensors
        s = torch.Tensor(s).to(self.device)
        a = torch.Tensor(a).to(self.device)
        r = torch.tensor(r, dtype=torch.float32)

        # Calculate loss
        logits = self._model(s)
        action_probabilities = nn.functional.softmax(logits, dim=1)
        selected_action_probabilities = torch.sum(action_probabilities * a, dim=1)
        
        # Calculate the policy gradient loss
        pg_loss = -torch.mean(torch.log(selected_action_probabilities + 1e-8) * r)
        
        # Calculate entropy to encourage exploration
        entropy = -torch.mean(torch.sum(action_probabilities * torch.log(action_probabilities + 1e-8), dim=1))
        
        # Total loss combines policy gradient loss and entropy regularization
        loss = pg_loss - beta * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()
    


class AdvantageActorCriticAgent(PolicyGradientAgent):
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        super().__init__(board_size=board_size, frames=frames,
                         buffer_size=buffer_size, gamma=gamma,
                         n_actions=n_actions, use_target_net=use_target_net,
                         version=version)
        self.optimizer = optim.RMSprop(self._model.parameters(), lr=5e-4)

    def _agent_model(self):
        input_channels = self._n_frames
        input_dim = self._board_size
        hidden_units = 64

        actor = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (input_dim - 4) * (input_dim - 4), hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, self._n_actions)
        )
        
        critic = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (input_dim - 4) * (input_dim - 4), hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )

        return actor, nn.Sequential(actor, critic), critic

    def reset_models(self):
        self._actor_model, self._full_model, self._values_model = self._agent_model()
        if self._use_target_net:
            self._target_net, _, _ = self._agent_model()
            self.update_target_net()

    def save_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        torch.save(self._actor_model.state_dict(), f"{file_path}/model_{iteration:04d}.pt")
        torch.save(self._full_model.state_dict(), f"{file_path}/model_{iteration:04d}_full.pt")
        if self._use_target_net:
            torch.save(self._values_model.state_dict(), f"{file_path}/model_{iteration:04d}_values.pt")
            torch.save(self._target_net.state_dict(), f"{file_path}/model_{iteration:04d}_target.pt")

    def load_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._actor_model.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}.pt"))
        self._full_model.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}_full.pt"))
        if self._use_target_net:
            self._values_model.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}_values.pt"))
            self._target_net.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}_target.pt"))

    def update_target_net(self):
        if self._use_target_net:
            self._target_net.load_state_dict(self._values_model.state_dict())

    def train_agent(self, batch_size=32, beta=0.001, normalize_rewards=False,
                    num_games=1, reward_clip=False):
        s, a, r, next_s, done, _ = self._buffer.sample(self._buffer.get_current_size())
        s_prepared = self._prepare_input(s)
        next_s_prepared = self._prepare_input(next_s)

        if normalize_rewards:
            if (r == r[0][0]).sum() == r.shape[0]:
                r -= r
            else:
                r = (r - np.mean(r)) / np.std(r)

        if reward_clip:
            r = np.sign(r)

        if self._use_target_net:
            next_s_pred = self._target_net(torch.Tensor(next_s_prepared).to(self.device))
        else:
            next_s_pred = self._values_model(torch.Tensor(next_s_prepared).to(self.device))
        s_pred = self._values_model(torch.Tensor(s_prepared).to(self.device))

        future_reward = self._gamma * next_s_pred * (1 - done)
        advantage = a * (r + future_reward - s_pred)
        critic_target = r + future_reward

        actor_logits, critic_values = self._full_model(torch.Tensor(s_prepared).to(self.device))
        action_probabilities = nn.functional.softmax(actor_logits, dim=1)
        log_policy = nn.functional.log_softmax(actor_logits, dim=1)

        J = torch.sum(advantage * log_policy) / num_games
        entropy = -torch.sum(action_probabilities * log_policy) / num_games
        actor_loss = -J - beta * entropy
        critic_loss = nn.functional.smooth_l1_loss(critic_values, torch.Tensor(critic_target).to(self.device))
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss = [loss.item(), actor_loss.item(), critic_loss.item()]
        return loss[0] if len(loss) == 1 else loss
    



class HamiltonianCycleAgent(Agent):
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=False,
                 version=''):
        assert board_size % 2 == 0, "Board size should be odd for hamiltonian cycle"
        super().__init__(board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        self._get_cycle_square()
    
    def _get_neighbors(self, point):
        row, col = point // self._board_size, point % self._board_size
        neighbors = []
        for delta_row, delta_col in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
            new_row, new_col = row + delta_row, col + delta_col
            if (1 <= new_row and new_row <= self._board_size - 2 and
                1 <= new_col and new_col <= self._board_size - 2):
                neighbors.append(new_row * self._board_size + new_col)
        return neighbors

    def _hamil_util(self):
        neighbors = self._get_neighbors(self._cycle[self._index])
        if (self._index == ((self._board_size - 2) ** 2) - 1):
            if (self._start_point in neighbors):
                return True
            else:
                return False
        else:
            for i in neighbors:
                if (i not in self._cycle_set):
                    self._index += 1
                    self._cycle[self._index] = i
                    self._cycle_set.add(i)
                    ret = self._hamil_util()
                    if (ret):
                        return True
                    else:
                        self._cycle_set.remove(self._cycle[self._index])
                        self._index -= 1
            return False

    def _get_cycle_square(self):
        self._cycle = np.zeros(((self._board_size - 2) ** 2,), dtype=np.int64)
        index = 0
        sp = 1 * self._board_size + 1
        while (index < self._cycle.shape[0]):
            if (index == 0):
                pass
            elif ((sp // self._board_size) == 2 and (sp % self._board_size) == self._board_size - 2):
                sp = ((sp // self._board_size) - 1) * self._board_size + (sp % self._board_size)
            elif (index != 1 and sp // self._board_size == 1):
                sp = ((sp // self._board_size)) * self._board_size + ((sp % self._board_size) - 1)
            elif ((sp % self._board_size) % 2 == 1):
                sp = ((sp // self._board_size) + 1) * self._board_size + (sp % self._board_size)
                if (sp // self._board_size == self._board_size - 1):
                    sp = ((sp // self._board_size) - 1) * self._board_size + ((sp % self._board_size) + 1)
            else:
                sp = ((sp // self._board_size) - 1) * self._board_size + (sp % self._board_size)
                if (sp // self._board_size == 1):
                    sp = ((sp // self._board_size) + 1) * self._board_size + ((sp % self._board_size) + 1)
            self._cycle[index] = sp
            index += 1

    def move(self, board, legal_moves, values):
        cy_len = (self._board_size - 2) ** 2
        curr_head = np.sum(self._board_grid * (board[:, :, 0] == values['head']).reshape(self._board_size, self._board_size))
        index = 0
        while (1):
            if (self._cycle[index] == curr_head):
                break
            index = (index + 1) % cy_len
        prev_head = self._cycle[(index - 1) % cy_len]
        next_head = self._cycle[(index + 1) % cy_len]

        if (board[prev_head // self._board_size, prev_head % self._board_size, 0] == 0):
            if (next_head > curr_head):
                return 3
            else:
                return 1
        else:
            curr_head_row, curr_head_col = self._point_to_row_col(curr_head)
            prev_head_row, prev_head_col = self._point_to_row_col(prev_head)
            next_head_row, next_head_col = self._point_to_row_col(next_head)
            dx, dy = next_head_col - curr_head_col, -next_head_row + curr_head_row
            if (dx == 1 and dy == 0):
                return 0
            elif (dx == 0 and dy == 1):
                return 1
            elif (dx == -1 and dy == 0):
                return 2
            elif (dx == 0 and dy == -1):
                return 3
            else:
                return -1

    def get_action_proba(self, board, values):
        move = self.move(board, values)
        prob = [0] * self._n_actions
        prob[move] = 1
        return prob

    def _get_model_outputs(self, board=None, model=None):
        return [[0] * self._n_actions]

    def load_model(self, **kwargs):
        pass




class SupervisedLearningAgent(DeepQLearningAgent):
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        super().__init__(board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        # Define a softmax layer on top of the Q-network for classification
        self.model_action_out = nn.Sequential(
            nn.Softmax(dim=1)
        )
        # Create a model that includes the Q-network and the softmax layer
        self.model_action = nn.Sequential(
            self.model,
            self.model_action_out
        )
        # Define the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model_action.parameters(), lr=0.0005)

    def train_agent(self, batch_size=32, num_games=1, epochs=5, 
                    reward_clip=False):
        s, a, _, _, _, _ = self.buffer.sample(self.get_buffer_size())
        # Convert data to PyTorch tensors
        s = torch.tensor(self.normalize_board(s), dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        # Train the model
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model_action(s)
            loss = self.criterion(outputs, a)
            loss.backward()
            self.optimizer.step()
        loss = round(loss.item(), 5)
        return loss

    def get_max_output(self):
        s, _, _, _, _, _ = self.buffer.sample(self.get_buffer_size())
        max_value = np.max(np.abs(self.model_action(torch.tensor(self.normalize_board(s), dtype=torch.float32)).detach().numpy()))
        return max_value

    def normalize_layers(self, max_value=None):
        if max_value is None or np.isnan(max_value):
            max_value = 1.0
        # Normalize the output layer by dividing by max_value
        output_layer_weights = list(self.model[1].parameters())
        output_layer_weights[0].data /= max_value
        output_layer_weights[1].data /= max_value


class BreadthFirstSearchAgent(Agent):
    def _get_neighbors(self, point, values, board):
        row, col = self._point_to_row_col(point)
        neighbors = []
        for delta_row, delta_col in [[-1,0], [1,0], [0,1], [0,-1]]:
            new_row, new_col = row + delta_row, col + delta_col
            if(board[new_row][new_col] in [values['board'], values['food'], values['head']]):
                neighbors.append(new_row*self._board_size + new_col)
        return neighbors

    def _get_shortest_path(self, board, values):
        board = board[:,:,0]
        head = ((self._board_grid * (board == values['head'])).sum())
        points_to_search = deque()
        points_to_search.append(head)
        path = []
        row, col = self._point_to_row_col(head)
        distances = np.ones((self._board_size, self._board_size)) * np.inf
        distances[row][col] = 0
        visited = np.zeros((self._board_size, self._board_size))
        visited[row][col] = 1
        found = False
        while(not found):
            if(len(points_to_search) == 0):
                path = []
                break
            else:
                curr_point = points_to_search.popleft()
                curr_row, curr_col = self._point_to_row_col(curr_point)
                n = self._get_neighbors(curr_point, values, board)
                if(len(n) == 0):
                    continue
                for p in n:
                    row, col = self._point_to_row_col(p)
                    if(distances[row][col] > 1 + distances[curr_row][curr_col]):
                        distances[row][col] = 1 + distances[curr_row][curr_col]
                    if(board[row][col] == values['food']):
                        found = True
                        break
                    if(visited[row][col] == 0):
                        visited[curr_row][curr_col] = 1
                        points_to_search.append(p)
        curr_point = ((self._board_grid * (board == values['food'])).sum())
        path.append(curr_point)
        while(1):
            curr_row, curr_col = self._point_to_row_col(curr_point)
            if(distances[curr_row][curr_col] == np.inf):
                return []
            if(distances[curr_row][curr_col] == 0):
                break
            n = self._get_neighbors(curr_point, values, board)
            for p in n:
                row, col = self._point_to_row_col(p)
                if(distances[row][col] != np.inf and \
                   distances[row][col] == distances[curr_row][curr_col] - 1):
                    path.append(p)
                    curr_point = p
                    break
        return path

    def move(self, board, legal_moves, values):
        if(board.ndim == 3):
            board = board.reshape((1,) + board.shape)
        board_main = board.copy()
        a = np.zeros((board.shape[0],), dtype=np.uint8)
        for i in range(board.shape[0]):
            board = board_main[i,:,:,:]
            path = self._get_shortest_path(board, values)
            if(len(path) == 0):
                a[i] = 1
                continue
            next_head = path[-2]
            curr_head = (self._board_grid * (board[:,:,0] == values['head'])).sum()
            if(((board[:,:,0] == values['head']) + (board[:,:,0] == values['snake']) \
                == (board[:,:,1] == values['head']) + (board[:,:,1] == values['snake'])).all()):
                prev_head = curr_head - 1
            else:
                prev_head = (self._board_grid * (board[:,:,1] == values['head'])).sum()
            curr_head_row, curr_head_col = self._point_to_row_col(curr_head)
            prev_head_row, prev_head_col = self._point_to_row_col(prev_head)
            next_head_row, next_head_col = self._point_to_row_col(next_head)
            dx, dy = next_head_col - curr_head_col, -next_head_row + curr_head_row
            if(dx == 1 and dy == 0):
                a[i] = 0
            elif(dx == 0 and dy == 1):
                a[i] = 1
            elif(dx == -1 and dy == 0):
                a[i] = 2
            elif(dx == 0 and dy == -1):
                a[i] = 3
            else:
                a[i] = 0
        return a

    def get_action_proba(self, board, values):
        move = self.move(board, values)
        prob = [0] * self._n_actions
        prob[move] = 1
        return prob

    def _get_model_outputs(self, board=None, model=None):
        return [[0] * self._n_actions]

    def load_model(self, **kwargs):
        pass