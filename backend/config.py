# Configuration for distributed LLM setup
# Update these settings based on your network

# GPU Node (Windows with NVIDIA GPU) - Server
GPU_NODE_HOST = "0.0.0.0"  # Listen on all interfaces
GPU_NODE_PORT = 50051

# CPU Node (Mac M4) - Client
CPU_NODE_HOST = "localhost"  # Change to GPU node's IP when running on Mac
CPU_NODE_PORT = 8080

# Network Configuration
# When running on Mac M4, change GPU_NODE_ADDRESS to your Windows machine's IP
# Example: "192.168.1.50:50051" (replace with actual Windows IP)
GPU_NODE_ADDRESS = "10.0.0.196:50051"  # Update this to your Windows machine's IP

# Model Configuration
MODEL_ID = "mistralai/Mistral-7B-v0.1"

# Layer Distribution (65-35 split)
# GPU Node: layers 21-32 (11 layers, ~65% of computation)
# CPU Node: layers 0-20 (21 layers, ~35% of computation)
GPU_LAYER_START = 21
GPU_LAYER_END = 32
CPU_LAYER_START = 0
CPU_LAYER_END = 21

# Memory optimization
TORCH_DTYPE = "float16"
LOW_CPU_MEM_USAGE = True 