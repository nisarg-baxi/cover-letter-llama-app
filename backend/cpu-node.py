from flask import Flask, request, Response, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import psutil
import time
from flask_cors import CORS
import gc
import grpc
import llm_pb2
import llm_pb2_grpc

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])

# GPU node configuration - Update this to your Mac M4's IP address
GPU_NODE_ADDRESS = "192.168.1.100:50051"  # Replace with your Mac M4's actual IP address

# Model setup
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)

# Load only layers 0-20 on CPU using device_map for 65-35 distribution
layer_start = 0   # Start from layer 0 (35% split)
layer_end = 21    # End at layer 20 (Mac handles first 21 layers)

# Create a custom device map that only loads specific layers to CPU
device_map = {}
device_map["model.embed_tokens"] = "cpu"  # Keep embedding on CPU
device_map["model.norm"] = "cpu"  # Keep norm on CPU
device_map["lm_head"] = "cpu"  # Keep lm_head on CPU

# Map only layers 0-20 to CPU (35% of layers)
for i in range(layer_start, layer_end):
    device_map[f"model.layers.{i}"] = "cpu"

# Map all other layers to CPU (they won't be loaded)
for i in range(32):  # Mistral-7B has 32 layers
    if i < layer_start or i >= layer_end:
        device_map[f"model.layers.{i}"] = "cpu"

print(f"Loading layers {layer_start} to {layer_end-1} on CPU (35% distribution)...")

full_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Extract only the layers we need
embedding = full_model.model.embed_tokens
first_layers = torch.nn.ModuleList()
for i in range(layer_start, layer_end):
    first_layers.append(full_model.model.layers[i])

hidden_dim = embedding.embedding_dim
print(f"Loaded embedding and layers {layer_start} to {layer_end-1} on CPU")

print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")

def try_gpu_forward(input_ids, attention_mask, position_ids):
    try:
        channel = grpc.insecure_channel(GPU_NODE_ADDRESS)
        stub = llm_pb2_grpc.LLMServiceStub(channel)
        request = llm_pb2.ForwardRequest(
            input_ids=input_ids[0].tolist(),
            attention_mask=attention_mask[0].tolist(),
            position_ids=position_ids[0].tolist()
        )
        response = stub.Forward(request, timeout=30)
        if response.activations:
            return torch.tensor(response.activations).reshape(input_ids.shape[0], -1)
        else:
            print("GPU node returned no activations.")
            return None
    except Exception as e:
        print(f"Error contacting GPU node: {e}")
        return None

@app.route('/generate', methods=['POST'])
def generate_cover_letter():
    def generate(user_message=None):
        if user_message:
            prompt = f"[INST] {user_message.strip()} [/INST]"
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

            # Run embedding and first N layers on CPU
            with torch.no_grad():
                hidden_states = embedding(input_ids)
                for layer in first_layers:
                    layer_output = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        output_attentions=False
                    )
                    hidden_states = layer_output[0]

            # Send activations to GPU node
            try:
                channel = grpc.insecure_channel(GPU_NODE_ADDRESS)
                stub = llm_pb2_grpc.LLMServiceStub(channel)
                request = llm_pb2.ForwardRequest(
                    activations=hidden_states.cpu().flatten().tolist(),
                    batch=hidden_states.shape[0],
                    seq_len=hidden_states.shape[1],
                    hidden_dim=hidden_states.shape[2],
                    attention_mask=attention_mask[0].tolist(),
                    position_ids=position_ids[0].tolist()
                )
                response = stub.Forward(request, timeout=60)
                if response.logits:
                    logits = torch.tensor(response.logits, dtype=torch.float32).reshape(
                        response.batch, response.seq_len, response.vocab_size
                    )
                    generated_ids = torch.argmax(logits, dim=-1)
                    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                else:
                    result = "[GPU node returned no logits]"
            except Exception as e:
                print(f"Error contacting GPU node: {e}")
                result = "[Error: GPU node unavailable]"

            for word in result.split():
                yield f"data: {word} \n\n"
                time.sleep(0.05)
        yield "data: \n\n"

    if request.method == 'POST':
        data = request.get_json()
        user_message = data.get('job_description', '') if data else ''
        return Response(generate(user_message), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/unload_model', methods=['POST'])
def unload_model():
    global full_model, tokenizer
    full_model = None
    tokenizer = None
    gc.collect()
    print("Model and tokenizer unloaded from memory.")
    return jsonify({"status": "unloaded"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)