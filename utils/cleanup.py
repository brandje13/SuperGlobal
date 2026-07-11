import torch
import gc
import sys


def print_vram_usage(step_name: str) -> None:
    # Check if CUDA is available before querying memory to avoid runtime errors
    if not torch.cuda.is_available():
        print(f"[{step_name}] CUDA is not available.")
        return

    # Calculate allocated and reserved memory in Megabytes
    allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)

    print(f"[{step_name}] VRAM Allocated: {allocated_mb:.2f} MB | VRAM Reserved: {reserved_mb:.2f} MB")


def find_leaking_tensors() -> None:
    # Force garbage collection to clear out standard unreferenced objects
    gc.collect()

    # Empty the PyTorch cache to release unbound memory back to the GPU
    torch.cuda.empty_cache()

    tensor_count = 0
    total_memory_bytes = 0

    # Iterate through all active objects tracked by the Python garbage collector
    for obj in gc.get_objects():
        try:
            # Check if the object is a PyTorch tensor or a Parameter holding a tensor
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                # Isolate tensors that reside on the GPU
                if obj.is_cuda:
                    tensor_count += 1

                    # Calculate memory footprint of the individual tensor
                    element_size = obj.element_size()
                    numel = obj.nelement()
                    tensor_size_bytes = element_size * numel
                    total_memory_bytes += tensor_size_bytes

                    print(f"CUDA Tensor Found - Size: {list(obj.size())}, "
                          f"Type: {obj.dtype}, Memory: {tensor_size_bytes / (1024 ** 2):.2f} MB")
        except Exception:
            # Ignore objects that raise exceptions when their properties are queried
            pass

    total_memory_mb = total_memory_bytes / (1024 ** 2)
    print(f"Total active CUDA tensors in memory: {tensor_count}")
    print(f"Total identifiable VRAM held by active tensors: {total_memory_mb:.2f} MB")