import functools

import torch


def pytorch_profile(
    enable=True,
    output_file=None,
    use_cuda=None,
    record_shapes=False,
    profile_memory=False,
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enable:
                return func(*args, **kwargs)

            use_cuda_prof = torch.cuda.is_available() if use_cuda is None else use_cuda
            try:
                with torch.autograd.profiler.profile(
                    use_cuda=use_cuda_prof,
                    record_shapes=record_shapes,
                    profile_memory=profile_memory,
                ) as prof:
                    result = func(*args, **kwargs)

                if output_file:
                    prof.export_chrome_trace(output_file)
                else:
                    print(
                        prof.key_averages().table(
                            sort_by="cuda_time_total", row_limit=10
                        )
                    )
                return result
            except Exception as e:
                raise e

        return wrapper

    return decorator


# # Example Usage
# @pytorch_profile(
#     enable=True,
#     output_file="profile_trace.json",
#     use_cuda=True,
#     record_shapes=True,
#     profile_memory=True,
# )
# def example_function(tensor):
#     return tensor + 10


# # Running the decorated function
# example_function(torch.rand(10, 10))
