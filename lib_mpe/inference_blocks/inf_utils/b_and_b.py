import os
import subprocess
import time

import numpy as np


def run_cpp_program_gpu(
    program_path,
    args,
    output_file="cpp_program_output.txt",
    max_time=60,  # Maximum runtime in seconds
):
    command = f"{program_path} {' '.join(args)}"

    try:
        # Open the output file in write mode
        with open(output_file, "w") as file:
            # Start the process using shell execution and redirect stdout to the file
            process = subprocess.Popen(
                command, stdout=file, stderr=subprocess.PIPE, shell=True
            )

            try:
                start_time = time.time()
                while True:
                    if process.poll() is not None:
                        break  # Process has finished
                    if time.time() - start_time > max_time:
                        process.kill()  # Kill the process on timeout
                        process.wait()
                        raise subprocess.TimeoutExpired(command, max_time)
                    time.sleep(1)
            except Exception as e:
                process.kill()
                process.wait()
                raise e
            finally:
                if process.stderr:
                    process.stderr.close()

        # Reopen the output file to check if the output has content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            with open(output_file, "r") as file:
                output = file.read()
            return output
        else:
            return "No output generated."
    except subprocess.TimeoutExpired:
        print(
            f"The program did not finish within {max_time} seconds and was terminated."
        )
        # Read partial output from file if available
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            with open(output_file, "r") as file:
                output = file.read()
            return output
        else:
            return "No output due to timeout."


def process_output(output, num_vars):
    for line in output.splitlines():
        if line.startswith("s "):
            parts = line.split()
            ll_score = float(parts[1])  # First value after 's'
            mpe_outputs = np.array(
                parts[3:], dtype=float
            )  # Remaining values as a numpy array
            return ll_score, mpe_outputs
    # If optimal solution is not found, get the best solution
    best_score = float("-inf")
    best_assignment = np.zeros(num_vars)

    lines = output.split("\n")  # Split the output into lines
    for line in lines:
        parts = line.split()  # Split each line into parts
        if len(parts) > 2 and parts[1] == "u":
            try:
                score = float(parts[4])  # Assuming the score is the fifth element
                assignment = parts[6:]  # The rest of the elements are the assignment
                if score > best_score:
                    best_score = score
                    best_assignment = assignment
            except ValueError:
                # Handle the case where conversion to float fails
                continue
    print(f"Best score: {best_score}")
    return best_score, np.array(best_assignment, dtype=float)


# Function to process each job
def run_branch_and_bound(
    idx,
    data_point,
    evid_vars_idx,
    temp_folder,
    problem_uai_file,
    i_bound=10,
    program_path="",
):
    temp_folder_path = temp_folder.name
    num_evid_vars = len(evid_vars_idx)
    num_vars = len(data_point)
    exp_path = os.path.join(temp_folder_path, "problems")
    output_file = os.path.join(temp_folder_path, f"output_{idx}.txt")
    os.makedirs(exp_path, exist_ok=True)

    problem_evid_file = os.path.join(exp_path, f"{idx}.uai.evid")
    evid_string = f"{num_evid_vars} \n" + "".join(
        [
            f"{evid_var_idx} {int(data_point[evid_var_idx])} \n"
            for evid_var_idx in evid_vars_idx
        ]
    )
    with open(problem_evid_file, "w") as f:
        f.write(evid_string)

    # default arguments
    program_args = [
        "-f",
        problem_uai_file,
        "-e",
        problem_evid_file,
        "-m",
        "4000",
        "-i",
        f"{i_bound}",
    ]

    output = run_cpp_program_gpu(program_path, program_args, output_file)
    best_score, solution = process_output(output, num_vars)

    return best_score, solution
