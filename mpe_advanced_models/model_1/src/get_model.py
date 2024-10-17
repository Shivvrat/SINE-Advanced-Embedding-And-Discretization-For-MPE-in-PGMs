import copy

from loguru import logger
from torch import nn

from lib_mpe.models.encoder.hgnn import HGNNEmbeddingLayer
from lib_mpe.models.head.nn import NeuralNetwork
from lib_mpe.models.head.transformer import TransformerEncoder
from lib_mpe.models.main_model import CombinedModel
from mpe_advanced_models.model_1.src.utils_folder.optim import select_optimizer


def init_model_and_optimizer(
    cfg,
    device,
    fabric,
    num_ip_features,
    num_pgm_feature,
    num_variables,
    run_type="train",
    **kwargs,
):
    num_layers = cfg.model_layers
    combined_model = init_model_and_embeddings(
        cfg,
        device,
        num_ip_features,
        num_pgm_feature,
        num_variables,
        kwargs,
        num_layers,
        is_teacher=False,
    )

    # Define the optimizer
    if run_type == "train":
        optimizer_name = cfg.train_optimizer
        lr = cfg.train_lr
        weight_decay = cfg.train_weight_decay
    else:
        optimizer_name = cfg.test_optimizer
        lr = cfg.test_lr
        weight_decay = cfg.test_weight_decay
    logger.info(f"We are in {run_type} mode")
    logger.info(f"Optimizer: {optimizer_name}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Weight decay: {weight_decay}")
    optimizer = select_optimizer(combined_model, optimizer_name, lr, weight_decay)
    combined_model, optimizer = fabric.setup(combined_model, optimizer)
    return combined_model, optimizer


def init_model_and_embeddings(
    cfg,
    device,
    num_ip_features,
    num_pgm_feature,
    num_variables,
    kwargs,
    num_layers,
    is_teacher=False,
):

    ip_embedding_size, embedding_layer = init_embeddings(cfg, device, num_ip_features)
    if cfg.embedding_type == "discrete":
        predefined_layers = [128 * (2**i) for i in range(num_layers)]
    elif cfg.embedding_type == "hgnn":
        last_layer = (num_variables // 256 + 1) * 256 + 128
        predefined_layers = [last_layer * (2**i) for i in reversed(range(num_layers))]
    if cfg.model in ["nn"]:
        hidden_size = predefined_layers
        pgm_embedding_size = num_pgm_feature
        if cfg.model == "nn":
            model = init_nn_model(
                cfg,
                device,
                num_variables,
                kwargs,
                hidden_size,
                ip_embedding_size,
                pgm_embedding_size,
                is_teacher=is_teacher,
            )
    elif cfg.model == "transformer":
        hidden_size = predefined_layers
        if cfg.take_global_mean_pool_embedding:
            raise ValueError("Global mean pooling not supported for transformer model")
        if cfg.embedding_type not in ["gnn", "hgnn"]:
            raise ValueError("Transformer model only supports gnn and hgnn embeddings")
        cfg.transformer_max_seq_length = num_ip_features
        cfg.transformer_d_model = ip_embedding_size // num_ip_features
        cfg.transformer_n_head = 4
        cfg.transformer_num_layers = cfg.transformer_layers
        cfg.transformer_dim_feedforward = 2048
        cfg.transformer_dropout = 0.1
        model = TransformerEncoder(
            cfg=cfg,
            dropout=cfg.dropout_rate,
            output_size=num_variables,
            output_layers_sizes=hidden_size,
        ).to(device)
    combined_model = CombinedModel(embedding_layer, model).to(device)
    return combined_model


def init_embeddings(cfg, device, num_ip_features):
    embeddings = None
    if cfg.embedding_type == "discrete":
        ip_embedding_size = num_ip_features * 2
    elif cfg.embedding_type in [
        "hgnn",
    ]:
        encoder_channels = list(map(int, cfg.encoder_embedding_dim.split(",")))
        encoder_channels = encoder_channels
        embeddings = HGNNEmbeddingLayer(
            num_ip_features,
            encoder_channels,
            cfg.initialize_embed_size,
            cfg.max_clique_size,
            cfg,
            device,
        )
        if cfg.take_global_mean_pool_embedding:
            ip_embedding_size = embeddings.embedding_size
        else:
            ip_embedding_size = embeddings.embedding_size * num_ip_features
    elif cfg.embedding_type == "gnn":
        raise NotImplementedError("GNN embedding not implemented")
    else:
        raise ValueError("Invalid embedding type")
    return ip_embedding_size, embeddings


def init_nn_model(
    cfg,
    device,
    num_variables,
    kwargs,
    hidden_size,
    ip_embedding_size,
    pgm_embedding_size,
    is_teacher,
):
    if cfg.model_type == "2":
        model = NeuralNetwork(
            cfg,
            ip_embedding_size + pgm_embedding_size,
            hidden_size,
            num_variables,
        ).to(device)
    else:
        raise ValueError("Invalid model type")
    return model
