import torch

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

