import onnxruntime
import sys

print("Python version:", sys.version)
print("ONNX Runtime version:", onnxruntime.__version__)
print("Available providers:", onnxruntime.get_available_providers())

try:
    # Try to verify if CUDA provider can actually be initialized
    # We don't have a model, but we can check if the shared library loads
    import ctypes
    # Try to load the cuda provider dll explicitly to check dependencies if possible, 
    # but onnxruntime loads it dynamically.
    
    # Just checking providers list usually implies they are registered, but not necessarily loadable without error during session creation
    # Let's try to create a session with a dummy model if possible? 
    # Or just rely on the fact that if it's in the list it MIGHT work, but the user error happened during session init.
    pass
except Exception as e:
    print(e)
