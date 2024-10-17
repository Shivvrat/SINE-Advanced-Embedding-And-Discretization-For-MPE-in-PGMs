import csv
import os


def construct_header(cfg):
    """
    Constructs the header row for the CSV file.
    """
    basic_headers = [
        "Run ID",
        "Mean LL NN",
        "Mean LL SPN",
        "Std LL NN",
        "Std LL SPN",
        "Outputs Path",
        "Runtime",
    ]
    arg_headers = [
        "Learning Rate",
        "Epochs",
        "Activation Function",
        "Optimizer",
        "LR Scheduler",
        "Input Type",
    ]
    return basic_headers + arg_headers


def construct_data_row(run_id, metrics, outputs_path, runtime, cfg):
    """
    Constructs a data row with run information and argument values.
    """
    basic_data = [
        run_id,
        metrics["mean_ll_nn"],
        metrics["mean_ll_pgm"],
        metrics["std_ll_nn"],
        metrics["std_ll_pgm"],
        outputs_path,
        runtime,
    ]
    params = [
        cfg.train_lr,
        cfg.epochs,
        cfg.activation_function,
        cfg.train_optimizer,
        cfg.lr_scheduler,
        cfg.input_type,
    ]
    return basic_data + params


def write_to_csv(csv_path, header, data_row):
    """
    Writes a header and a data row to a CSV file.
    """
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data_row)
