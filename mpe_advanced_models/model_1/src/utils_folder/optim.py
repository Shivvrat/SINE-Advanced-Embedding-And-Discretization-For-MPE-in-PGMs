import torch.optim as optim
from loguru import logger


def select_optimizer(model, optimizer_name, lr, weight_decay, for_teacher=False):
    """
    The function `select_optimizer` takes in a model and arguments, and returns the specified optimizer
    based on the optimizer name provided in the arguments.

    :param model: The `model` parameter is the neural network model that you want to optimize. It should
    be an instance of a PyTorch `nn.Module` subclass
    :param cfg: The `cfg` parameter is an object or dictionary that contains the values for the
    optimizer parameters. It should have the following attributes:
    :return: an optimizer object based on the optimizer name provided as an argument.
    """
    if for_teacher:
        logger.info(f"Teacher optimizer: {optimizer_name}")
        logger.info(
            f"We do not use weight decay for the teacher model. Weight decay: 0"
        )
        weight_decay = 0
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # elif optimizer_name == "lbfgs":
    #     optimizer = optim.LBFGS(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif optimizer_name == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=lr)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer name")
    return optimizer
