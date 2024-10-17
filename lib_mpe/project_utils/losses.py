import torch

l2_loss = torch.nn.MSELoss()


def distance_loss(pgm, output_for_pgm, initial_data, buckets, evid_lambda):
    output_for_pgm_distance = output_for_pgm.clone()
    evidence_true = initial_data.detach().clone()

    nan_value = float("nan")
    for indices in [buckets["query"], buckets["unobs"]]:
        output_for_pgm_distance[indices] = nan_value
        evidence_true[indices] = nan_value

    nn_evid_ll = pgm.evaluate(output_for_pgm_distance)
    with torch.no_grad():
        pgm_evid_ll = pgm.evaluate(evidence_true)

    evid_distance_loss = l2_loss(nn_evid_ll, pgm_evid_ll)
    return evid_lambda * evid_distance_loss


def evid_loss(output, evid_bucket, initial_data, evid_lambda):
    evidence_output = output[evid_bucket]
    evidence_true = initial_data[evid_bucket]
    evid_loss_value = l2_loss(evidence_output, evidence_true)
    return evid_lambda * evid_loss_value


def entropy_loss_function(predictions, lambda_param=0.01, epsilon=1e-5):
    """
    The entropy_loss_function calculates the entropy loss of a set of predictions with an optional
    regularization term.

    """
    regularizer = -(
        predictions * torch.log(predictions + epsilon)
        + (1 - predictions) * torch.log(1 - predictions + epsilon)
    )
    regularizer = lambda_param * regularizer.mean()
    return regularizer


def entropy_loss(output, entropy_lambda):
    entropy_loss_value = entropy_loss_function(output, entropy_lambda)
    return entropy_loss_value


def calculate_kl_divergence(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
