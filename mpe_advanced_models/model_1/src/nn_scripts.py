import copy

import numpy as np
import torch
from data.dataloaders import (create_buckets_all_queries_per,
                              create_buckets_one_query_per)
from get_model import init_model_and_optimizer
from loguru import logger
# from project_utils.profiling import pytorch_profile
from torch import nn
from tqdm import tqdm

import wandb


def get_buckets_for_iter(cfg, num_var_in_buckets, batch_data):
    sample_size, num_vars = batch_data["initial_data"].shape
    if cfg.use_single_model:
        single_example_bucket = create_buckets_all_queries_per(
            sample_size, cfg.task, batch_data["initial_data"].device
        )
    else:
        single_example_bucket = create_buckets_one_query_per(
            num_vars,
            num_var_in_buckets,
            batch_data["initial_data"].device,
        )

    # check if the bucket is empty - if so, then create a bucket with all False values
    for bucket_name in ["evid", "query", "unobs"]:
        single_example_bucket.setdefault(
            bucket_name, torch.zeros(sample_size, dtype=torch.bool)
        )
        # duplicate each value in the dict single_example_bucket to create a dict with batch_data["initial"].shape[0] duplicates for each key in torch
    single_example_bucket = {
        key: torch.stack(
            [single_example_bucket[key]] * batch_data["initial_data"].shape[0]
        )
        for key in single_example_bucket.keys()
    }

    return single_example_bucket


def pre_process_data(cfg, num_var_in_buckets, data_pack, train=True):
    # Check if cfg.same_bucket_epoch is true - if so, then use the same bucket for all the data points in the epoch
    if train:
        if cfg.same_bucket_iter:
            # create same bucket for all the data points in the epoch
            single_example_bucket = get_buckets_for_iter(
                cfg, num_var_in_buckets, data_pack
            )
            # Update the batch_data with the new bucket
            data_pack["evid"] = single_example_bucket["evid"]
            data_pack["query"] = single_example_bucket["query"]
            data_pack["unobs"] = single_example_bucket["unobs"]
    return data_pack


def get_embedding(cfg, data_pack, embedding_layer):
    if embedding_layer is not None:
        data_pack = embedding_layer(data_pack)
    return data_pack


def train(
    cfg, model, pgm, device, fabric, train_loader, optimizer, epoch, schedular, **kwargs
):
    """
    The `train` function is used to train a model using a given dataset and optimizer, and logs the
    training loss.

    :param cfg: The `cfg` parameter is a dictionary or object that contains various arguments or
    configuration settings for the training process. It is used to pass information such as batch size,
    learning rate, number of epochs, etc
    :param model: The `model` parameter is the neural network model that you want to train. It should be
    an instance of a PyTorch model class
    :param pgm: The "pgm" parameter is likely referring to a Sum-Product Network (pgm) model. pgms are a
    type of probabilistic graphical model that can be used for various tasks such as classification,
    regression, and anomaly detection. In this context, the pgm model is being used for
    :param device: The `device` parameter is used to specify whether the model should be trained on a
    CPU or a GPU. It is typically a string value, such as "cpu" or "cuda:0", where "cuda:0" refers to
    the first available GPU
    :param fabric: The `fabric` parameter is a module that contains the pgm (Sum-Product Network)
    architecture. It is used to create and manipulate the pgm model
    :param train_loader: The `train_loader` parameter is a PyTorch `DataLoader` object that is used to
    load the training data in batches. It is responsible for iterating over the training dataset and
    providing batches of data to the model during training
    :param optimizer: The optimizer is an object that implements the optimization algorithm. It is used
    to update the parameters of the model during training. Examples of optimizers include Stochastic
    Gradient Descent (SGD), Adam, and RMSprop
    :param epoch: The `epoch` parameter represents the current epoch number during training. An epoch is
    a complete pass through the entire training dataset
    :return: the average training loss for the epoch.
    """
    model.train()
    train_loss = 0
    num_var_in_buckets = train_loader.dataset.num_var_in_buckets
    for batch_idx, (batch_data) in enumerate(train_loader):
        embedding_layer = model.get_embedding()
        optimizer.zero_grad()
        data_pack = pre_process_data(cfg, num_var_in_buckets, batch_data)
        data_pack = get_embedding(cfg, data_pack, embedding_layer)
        main_model = model.get_main_model()
        loss = main_model.train_iter(pgm, data_pack)
        # add gradient clipping if cfg.add_gradient_clipping is true
        if cfg.add_gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        fabric.backward(loss)
        optimizer.step()
        final_loss = loss
        train_loss += final_loss.item()
        if schedular is not None and cfg.lr_scheduler == "OneCycleLR":
            # cyclic learning rate scheduler needs to be called after every batch
            schedular.step()

        # logging
        if batch_idx % cfg.log_interval == 0:
            current_loss = loss.item()
            logger.info(
                f"Train Epoch: {epoch} [{batch_idx * len(data_pack['index'])}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\t Training Loss: {current_loss:.6f}"
            )

            if cfg.dry_run:
                break
    train_loss /= batch_idx + 1
    wandb.log({"Train": train_loss})
    logger.info("\Train set: Average loss: {:.4f}".format(train_loss))
    return train_loss


@torch.no_grad()
def validate(cfg, model, pgm, device, test_loader, best_loss, counter):
    model.eval()
    test_loss = 0
    min_delta = (
        0.001  # Minimum change in the validation loss to be considered as improvement
    )
    all_unprocessed_data = []
    all_nn_outputs = []
    all_outputs_for_pgm = []
    all_buckets = {"evid": [], "query": [], "unobs": []}
    num_var_in_buckets = None
    for batch_idx, (batch_data) in enumerate(test_loader):
        embedding_layer = model.get_embedding()
        data_pack = pre_process_data(cfg, num_var_in_buckets, batch_data)
        data_pack = get_embedding(cfg, data_pack, embedding_layer)
        # Compute validation loss

        main_model = model.get_main_model()
        loss = main_model.validate_iter(
            pgm,
            all_unprocessed_data,
            all_nn_outputs,
            all_outputs_for_pgm,
            all_buckets,
            data_pack,
        )
        test_loss += loss.item()
        if batch_idx % cfg.log_interval == 0:
            current_loss = loss.item()
            logger.info(
                f"Validation Epoch: [{batch_idx * len(data_pack['index'])}/{len(test_loader.dataset)} "
                f"({100.0 * batch_idx / len(test_loader):.0f}%)]\t Validations Loss: {current_loss:.6f}"
            )

    # Calculate average test loss
    test_loss /= batch_idx + 1
    logger.info(f"\nTest set: Average loss: {test_loss:.4f}")
    wandb.log({"validation_loss": test_loss})
    # Early stopping check and best loss update
    if test_loss < best_loss - min_delta:
        best_loss = test_loss  # Update best loss if improved
        counter = 0  # Reset counter on improvement
    else:
        counter += 1  # Increment counter on no improvement
    # All output for pgm is the output of the NN after adding the values of the evidence - which can be used to calculate log liklihood of the pgm
    return (
        best_loss,
        test_loss,
        counter,
        all_unprocessed_data,
        all_nn_outputs,
        all_outputs_for_pgm,
        all_buckets,
    )


def prepare_data_pack_standard(data_packs, device):
    return {
        key: (
            value.unsqueeze(1).to(device)
            if value is not None
            and (isinstance(value, torch.Tensor) or isinstance(value, np.ndarray))
            else value
        )
        for key, value in data_packs.items()
    }


def prepare_data_pack_with_duplication(cfg, data_packs, idx):
    batch_size = min(cfg.test_batch_size, 512) if cfg.test_batch_size < 512 else 128
    return [
        (
            each[idx].unsqueeze(0).repeat(batch_size, 1)
            if each is not None
            and (isinstance(each, torch.Tensor) or isinstance(each, np.ndarray))
            else None
        )
        for each in data_packs
    ]


def update_loss_history(loss_history, loss, convergence_threshold):
    loss_history.append(loss)
    if len(loss_history) > 5:
        loss_history.pop(0)
        return all(
            abs(loss - loss_history[0]) < convergence_threshold for loss in loss_history
        )
    return False


def has_converged(
    loss_history,
    convergence_iter,
    loss,
    convergence_threshold,
    early_stopping_patience=5,
):
    if update_loss_history(loss_history, loss, convergence_threshold):
        convergence_iter += 1
        return convergence_iter, convergence_iter >= early_stopping_patience
    else:
        return 0, False


def perturb_weights(model, perturbation_level=0.05):
    """
    Perturbs the weights of the model slightly to help escape local minima.
    """
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.size()) * perturbation_level
            param.add_(noise)


def train_and_validate_single_example(
    cfg,
    model,
    pgm,
    fabric,
    loader,
    optimizer,
    best_loss,
    counter,
    device,
    num_data_features,
    num_pgm_feature,
    num_outputs,
    num_query_variables,
):
    """
    The `train_and_validate_single_example` function is used to train and validate a model on a given dataset.
    It returns the best loss, along with other outputs that can be used for further analysis.
    """
    all_unprocessed_data, all_nn_outputs, all_outputs_for_pgm = [], [], []
    all_buckets = {"evid": [], "query": [], "unobs": []}
    test_loss = 0
    # Save the initial model state outside the loop to avoid repeated deep copying
    embedding_layer = model.get_embedding()
    if embedding_layer is not None:
        embedding_layer_state = copy.deepcopy(embedding_layer.state_dict())
    main_model = model.get_main_model()
    model_state = copy.deepcopy(main_model.state_dict())
    num_var_in_buckets = loader.dataset.num_var_in_buckets
    dataset = loader.dataset[:]
    num_examples = len(dataset["initial_data"])
    if cfg.same_bucket_iter:
        cfg.same_bucket_iter = False
        logger.info("Same bucket iter is not allowed for train and validate")

    data_packs = pre_process_data(cfg, num_var_in_buckets, dataset)

    if not cfg.duplicate_example_train_on_test:
        data_packs = prepare_data_pack_standard(data_packs, device)

    if not cfg.only_test_train_on_test_set:
        assert (
            cfg.num_init_train_on_test == 1
        ), "When a model is trained, we should not take multiple initializations of the NN"

    convergence_threshold, perturbation_level = 1e-3, 0.05

    # go over each example in the dataset
    for idx in tqdm(range(num_examples)):
        # Initialize the optimizer for each new example
        if cfg.train_on_test_set_scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.num_iter_train_on_test // 5, gamma=0.8
            )
        if embedding_layer is not None:
            embedding_layer.load_state_dict(embedding_layer_state)
        main_model.load_state_dict(model_state)
        main_model.train()
        if embedding_layer is not None:
            embedding_layer.train()
        if cfg.duplicate_example_train_on_test:
            # make every tensor 2d
            data_pack = prepare_data_pack_with_duplication(cfg, data_packs, idx)
        else:
            data_pack = {
                key: (
                    data_packs[key][idx]
                    if data_packs[key] is not None
                    and (
                        isinstance(data_packs[key], torch.Tensor)
                        or isinstance(data_packs[key], np.ndarray)
                    )
                    else None
                )
                for key in data_packs.keys()
            }
        loss_history = []
        convergence_iter = 0
        for num_init in range(cfg.num_init_train_on_test):
            if cfg.only_test_train_on_test_set:
                # Initialize the weights of the model in for each new initialization
                main_model.initialize_weights()
            # train the model on the test set for a few iterations
            for iter in range(cfg.num_iter_train_on_test):
                optimizer.zero_grad()
                data_pack = get_embedding(cfg, data_pack, embedding_layer)
                loss = main_model.train_iter(pgm, data_pack)
                fabric.backward(loss)
                optimizer.step()
                if cfg.train_on_test_set_scheduler == "StepLR":
                    scheduler.step()
                convergence_iter, converged = has_converged(
                    loss_history,
                    convergence_iter,
                    loss.item(),
                    convergence_threshold,
                    cfg.early_stopping_patience,
                )
                if converged:
                    print("Convergence detected, stopping training")
                    break
                if cfg.debug and iter == cfg.num_iter_train_on_test - 1:
                    logger.info(f"Example {idx}, Iter {iter}, Loss {loss.item()}")
        with torch.no_grad():
            main_model.eval()
            if embedding_layer is not None:
                embedding_layer.eval()
            data_pack = get_embedding(cfg, data_pack, embedding_layer)

            loss = main_model.validate_iter(
                pgm,
                all_unprocessed_data,
                all_nn_outputs,
                all_outputs_for_pgm,
                all_buckets,
                data_pack,
            )
            test_loss += loss.item()
    # Average test loss calculation
    test_loss /= idx + 1
    logger.info(f"\nTest set: Average loss: {test_loss:.4f}")
    wandb.log({"test_loss": test_loss})
    # All output for pgm is the output of the NN after adding the values of the evidence - which can be used to calculate log likelihood of the pgm
    return (
        best_loss,
        test_loss,
        counter,
        all_unprocessed_data,
        all_nn_outputs,
        all_outputs_for_pgm,
        all_buckets,
    )
