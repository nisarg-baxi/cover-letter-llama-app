# Distributed LLM Setup Guide: Windows GPU + Mac M4

## Overview

This setup distributes Mistral-7B (8B parameters) between:

- **Windows GPU (4GB VRAM)**: Layers 21-32 (65% of computation)
- **Mac M4 (16GB RAM)**: Layers 0-20 (35% of computation)

## Network Setup

### 1. Find Your Windows Machine's IP Address

On Windows, run:

```cmd
ipconfig
```

Look for your local IP (usually starts with 192.168.x.x or 10.0.x.x)

### 2. Update Configuration

Edit `backend/config.py` and update:

```python
GPU_NODE_ADDRESS = "YOUR_WINDOWS_IP:50051"  # Replace with actual IP
```

### 3. Firewall Configuration

- **Windows**: Allow inbound connections on port 50051
- **Mac**: No special configuration needed (client)

## Running the System

### Step 1: Start GPU Node (Windows)

```bash
cd backend
python gpu_node.py
```

You should see:

```
Loading layers 21 to 31 on GPU (65% distribution)...
gRPC server running on port 50051...
```

### Step 2: Start CPU Node (Mac M4)

```bash
cd backend
python cpu-node.py
```

You should see:

```
Loading layers 0 to 20 on CPU (35% distribution)...
Flask server running on port 8080...
```

### Step 3: Test the System

Send a request to `http://localhost:8080/generate` with:

```json
{
  "job_description": "Write a cover letter for a software engineer position"
}
```

## Memory Distribution

### GPU Node (Windows)

- **Layers**: 21-32 (11 layers)
- **Memory**: ~2.6GB VRAM
- **Computation**: ~65% of model inference

### CPU Node (Mac M4)

- **Layers**: 0-20 (21 layers) + embedding + norm
- **Memory**: ~5.6GB RAM
- **Computation**: ~35% of model inference

## Troubleshooting

### Common Issues:

1. **Connection Refused**

   - Check Windows firewall settings
   - Verify IP address in config.py
   - Ensure both machines are on same network

2. **Out of Memory**

   - GPU: Try reducing batch size or sequence length
   - CPU: Close other applications to free RAM

3. **Model Loading Issues**
   - Ensure sufficient disk space for model download
   - Check internet connection for model download

### Performance Tips:

1. **Network**: Use wired connection for better performance
2. **Memory**: Close unnecessary applications
3. **Batch Size**: Start with batch_size=1 for testing

## Expected Performance

- **Latency**: ~2-5 seconds per request (depending on network)
- **Throughput**: ~10-20 requests per minute
- **Memory Usage**:
  - GPU: ~2.6GB VRAM
  - CPU: ~5.6GB RAM

## Next Steps

1. Test with different prompts
2. Monitor memory usage
3. Adjust layer distribution if needed
4. Consider quantization for even better memory efficiency
