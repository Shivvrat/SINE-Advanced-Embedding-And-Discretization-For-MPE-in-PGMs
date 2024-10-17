import torch
import torch.nn as nn
from loguru import logger

from lib_mpe.models.act_func.gumbel_softmax import GumbelSigmoid
from lib_mpe.project_utils.losses import *


class BaseHead(nn.Module):
    def __init__(self, cfg):
        """
        Initialize the BaseHead class.

        Parameters:
        - cfg: Configuration object containing model parameters.
        """
        super(BaseHead, self).__init__()
        self.cfg = cfg
        self.set_loss_bools(cfg)
        self.dropout = nn.Dropout(cfg.dropout_rate) if not cfg.no_dropout else None
        self.no_dropout = cfg.no_dropout
        self.device = cfg.device

    def set_loss_bools(self, cfg):
        """
        Set the loss booleans based on the configuration.

        Parameters:
        - cfg: Configuration object containing model parameters.
        """
        self.no_log_loss = cfg.no_log_loss
        self.add_distance_loss_evid_ll = cfg.add_distance_loss_evid_ll
        self.add_evid_loss = cfg.add_evid_loss
        self.evid_lambda = cfg.evid_lambda
        self.entropy_lambda = cfg.entropy_lambda
        self.add_entropy_loss = cfg.add_entropy_loss

    def process_buckets_single_row_for_pgm(
        self, nn_output, true, evid_bucket, query_bucket, unobs_bucket
    ):
        """
        Process the buckets for a single row for the PGM.

        Parameters:
        - nn_output: Neural network output.
        - true: True values.
        - evid_bucket: Evidence bucket.
        - query_bucket: Query bucket.
        - unobs_bucket: Unobserved bucket.
        """
        buckets = {"evid": evid_bucket, "query": query_bucket, "unobs": unobs_bucket}
        final_sample = nn_output.clone().requires_grad_(True)
        evid_indices = buckets["evid"]
        unobs_indices = buckets["unobs"]
        final_sample[evid_indices] = true[evid_indices]
        final_sample[unobs_indices] = float("nan")
        return buckets, final_sample

    def get_inputs(self, data_pack):
        """
        Get the inputs from the data pack.

        Parameters:
        - data_pack: Data pack containing the inputs.
        """
        return (
            data_pack["embedding"],
            data_pack["initial_data"],
            data_pack["evid"],
            data_pack["query"],
            data_pack["unobs"],
            data_pack["attention_mask"],
        )

    def process_model_output(self, embedding, test=False):
        """
        Process the model output.

        Parameters:
        - embedding: Embedding.
        - test: Test flag.
        """
        model_output = self(embedding)
        output = self._apply_activation(self.activation_function, model_output, test)
        if torch.isnan(output).any():
            logger.info(output)
            raise ValueError("Nan in output")
        return output

    def compute_loss(
        self, pgm, output_for_pgm, output, initial_data, buckets, return_mean
    ):
        """
        Compute the loss.

        Parameters:
        - pgm: PGM.
        - output_for_pgm: Output for the PGM.
        - output: Output.
        - initial_data: Initial data.
        - buckets: Buckets.
        - return_mean: Return mean flag.
        """
        loss = self._calculate_loss(pgm, output_for_pgm, output, initial_data, buckets)
        return loss.mean() if return_mean else loss

    def train_iter(self, pgm, data_pack, return_mean=True):
        """
        Train the model on a single iteration.

        Parameters:
        - pgm: PGM.
        - data_pack: Data pack containing the inputs.
        - return_mean: Return mean flag.
        """
        embedding, initial_data, evid_bucket, query_bucket, unobs_bucket, _ = (
            self.get_inputs(data_pack)
        )
        output = self.process_model_output(embedding, test=False)
        buckets, output_for_pgm = self.process_buckets_single_row_for_pgm(
            output, initial_data, evid_bucket, query_bucket, unobs_bucket
        )
        return self.compute_loss(
            pgm, output_for_pgm, output, initial_data, buckets, return_mean
        )

    def validate_iter(
        self,
        pgm,
        all_unprocessed_data,
        all_nn_outputs,
        all_outputs_for_pgm,
        all_buckets,
        data_pack,
        return_mean=True,
    ):
        """
        Validate the model on a single iteration.

        Parameters:
        - pgm: PGM.
        - all_unprocessed_data: All unprocessed data.
        - all_nn_outputs: All neural network outputs.
        - all_outputs_for_pgm: All outputs for the PGM.
        - all_buckets: All buckets.
        - data_pack: Data pack containing the inputs.
        - return_mean: Return mean flag.
        """
        embedding, initial_data, evid_bucket, query_bucket, unobs_bucket, _ = (
            self.get_inputs(data_pack)
        )
        output = self.process_model_output(embedding, test=True)
        buckets, output_for_pgm = self.process_buckets_single_row_for_pgm(
            output, initial_data, evid_bucket, query_bucket, unobs_bucket
        )
        loss = self.compute_loss(
            pgm, output_for_pgm, output, initial_data, buckets, return_mean
        )
        self.update_validation_data(
            all_unprocessed_data,
            all_nn_outputs,
            all_outputs_for_pgm,
            all_buckets,
            initial_data,
            output,
            output_for_pgm,
            buckets,
        )
        return loss

    def update_validation_data(
        self,
        all_unprocessed_data,
        all_nn_outputs,
        all_outputs_for_pgm,
        all_buckets,
        initial_data,
        output,
        output_for_pgm,
        buckets,
    ):
        """
        Update the validation data.

        Parameters:
        - all_unprocessed_data: All unprocessed data.
        - all_nn_outputs: All neural network outputs.
        - all_outputs_for_pgm: All outputs for the PGM.
        - all_buckets: All buckets.
        - initial_data: Initial data.
        - output: Output.
        - output_for_pgm: Output for the PGM.
        - buckets: Buckets.
        """
        all_nn_outputs.extend(output.detach().cpu().tolist())
        all_unprocessed_data.extend(initial_data.detach().cpu().tolist())
        all_outputs_for_pgm.extend(output_for_pgm.detach().cpu().tolist())
        for key in buckets:
            all_buckets[key].extend(buckets[key].detach().cpu().tolist())

    def _set_activation_func(self, cfg):
        """
        Set the activation function.

        Parameters:
        - cfg: Configuration object containing model parameters.
        """
        # Set hidden activation function
        self.hidden_activation_function = cfg.hidden_activation_function
        if cfg.hidden_activation_function == "relu":
            self.hidden_activation = nn.ReLU
        elif cfg.hidden_activation_function == "leaky_relu":
            self.hidden_activation = nn.LeakyReLU

        # Set output activation function
        self.activation_function = cfg.activation_function
        if cfg.activation_function == "sigmoid":
            self.activation = nn.Sigmoid()
        elif cfg.activation_function == "hard_sigmoid":
            self.activation = nn.Hardsigmoid()
        elif cfg.activation_function == "gumbel_sigmoid":
            self.activation = GumbelSigmoid(
                initial_temperature=cfg.gumbel_temperature,
                min_temperature=cfg.gumbel_min_temperature,
                annealing_rate=cfg.gumbel_annealing_rate,
            )
        else:
            raise ValueError(
                f"Unsupported activation function: {cfg.activation_function}"
            )

    def _apply_activation(self, activation_function, model_output, test=False):
        """
        Apply the activation function.

        Parameters:
        - activation_function: Activation function.
        - model_output: Model output.
        - test: Test flag.
        """
        if activation_function == "sigmoid":
            model_output = torch.sigmoid(model_output)
        elif activation_function == "hard_sigmoid":
            model_output = nn.Hardsigmoid()(model_output)
        elif activation_function == "gumbel_sigmoid":
            model_output = self.activation(model_output, hard=test)
        else:
            raise ValueError("Activation function not supported for NN model 2")
        return model_output

    def _calculate_loss(self, pgm, output_for_pgm, output, initial_data, buckets):
        """
        Calculate the loss.

        Parameters:
        - pgm: PGM.
        - output_for_pgm: Output for the PGM.
        - output: Output.
        - initial_data: Initial data.
        - buckets: Buckets.
        """
        query_bucket, evid_bucket = buckets["query"], buckets["evid"]
        final_func_value = pgm.evaluate(output_for_pgm)
        loss_from_pgm = (
            -final_func_value if not self.no_log_loss else -torch.exp(final_func_value)
        )
        loss = loss_from_pgm
        if self.add_distance_loss_evid_ll:
            loss += distance_loss(
                pgm, output_for_pgm, initial_data, buckets, self.evid_lambda
            )
        if self.add_evid_loss:
            loss += evid_loss(output, evid_bucket, initial_data, self.evid_lambda)
        if self.add_entropy_loss:
            loss += entropy_loss(output, self.entropy_lambda)
        return loss

    def update_output_layer(self, updated_output_layer):
        """
        Update the output layer.

        Parameters:
        - updated_output_layer: Updated output layer.
        """
        with torch.no_grad():
            self.output_layer.copy_(updated_output_layer)
