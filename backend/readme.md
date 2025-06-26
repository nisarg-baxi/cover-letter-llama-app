## Running the GPU Node (Windows)

1. Install dependencies:

   ```
   pip install -r backend/requirements.txt
   ```

2. Generate gRPC Python code:

   ```
   cd backend
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. llm.proto
   ```

3. (Optional) Pre-download the model:

   ```
   python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')"
   ```

4. Start the GPU node server:

   ```
   python gpu_node.py
   ```

5. Make sure your firewall allows inbound connections on port 50051.
