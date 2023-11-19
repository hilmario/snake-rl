import torch
import numpy as np
import pickle
from replay_buffer import ReplayBufferNumpy

def huber_loss(y_true, y_pred, delta=1):
    """
    PyTorch implementation of Huber loss.
    Parameters:
        y_true (Tensor): The true values for the regression data.
        y_pred (Tensor): The predicted values for the regression data.
        delta (float): The cutoff to decide whether to use quadratic or linear loss.
    Returns:
        Tensor: loss values for all points.
    """
    error = y_true - y_pred
    is_small_error = torch.abs(error) < delta
    squared_loss = torch.square(error) * 0.5
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)

    return torch.where(is_small_error, squared_loss, linear_loss).mean()

class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    HamiltonianCycleAgent
    BreadthFirstSearchAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """ initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                             .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row*self._board_size + col
    
    class DeepQLearningAgent(Agent):
        """
        This agent learns the game via Q learning.
        Model outputs everywhere refer to Q values.
        This class extends to the following classes:
        - PolicyGradientAgent
        - AdvantageActorCriticAgent
        """

    def __init__(self, board_size=10, frames=4, buffer_size=10000, gamma=0.99, n_actions=3, use_target_net=True, version=''):
        # Initialization can remain mostly the same I think
        super().__init__(board_size=board_size, frames=frames, buffer_size=buffer_size, gamma=gamma, n_actions=n_actions, use_target_net=use_target_net, version=version)
        self.reset_models()

    def reset_models(self):
        # Needs to be changed for PyTorch model initialization
        pass

    def _prepare_input(self, board):
        # Can remain the same I think
        if board.ndim == 3:
            board = board.reshape((1,) + self._input_shape)
        return self._normalize_board(board.copy())

    def _get_model_outputs(self, board, model=None):
        # Needs adaptation for PyTorch model inference
        pass

    def _normalize_board(self, board):
        # No change needed unless it involves TensorFlow-specific operations
        return board.astype(np.float32) / 4.0

    def move(self, board, legal_moves, value=None):
        # Adapt for PyTorch, particularly the model output processing
        pass

    def _agent_model(self):
        # Needs complete rewrite in PyTorch
        pass

    def set_weights_trainable(self):
        # Adapt for PyTorch's way of handling trainable weights
        pass

    def get_action_proba(self, board, values=None):
        # Adapt for PyTorch, especially the model output manipulation
        pass

    def save_model(self, file_path='', iteration=None):
        # Adapt for saving models in PyTorch format
        pass

    def load_model(self, file_path='', iteration=None):
        # Adapt for loading models in PyTorch format
        pass

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        # Significant changes needed for training in PyTorch
        pass

    def update_target_net(self):
        # Adapt for PyTorch's way of copying weights
        pass

    def compare_weights(self):
        # Change to compare weights in PyTorch models
        pass

    def copy_weights_from_agent(self, agent_for_copy):
        # Adapt for PyTorch's way of copying weights
        pass
