from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil

app = Flask(__name__)

# Model setup
model_id = "SweatyCrayfish/llama-3-8b-quantized"
tokenizer = AutoTokenizer.from_pretrained(model_id)

try:
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    embedding_layer = model.model.embed_tokens.to("cuda")  # Embedding layer
    total_layers = len(model.model.layers)  # Should be 32
    split_point = total_layers // 2  # 16
    gpu_layers = torch.nn.Sequential(*model.model.layers[:split_point]).to("cuda")
    print(f"Loaded embedding layer and {split_point} layers on GPU")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print(f"VRAM Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"RAM Available: {psutil.virtual_memory().available / 1024**3:.2f} GB")

@app.route('/gpu_forward', methods=['POST'])
def gpu_forward():
    try:
        data = request.get_json()
        print("Received input_ids:", data['input_ids'])
        input_ids = torch.tensor(data['input_ids']).to("cuda")
        print("Input shape:", input_ids.shape)
        
        with torch.no_grad():
            # Step 1: Convert input_ids to embeddings
            embeddings = embedding_layer(input_ids)  # Shape: [1, seq_len, 4096]
            print("Embeddings shape:", embeddings.shape)
            # Step 2: Pass through GPU layers
            activations = gpu_layers(embeddings)  # Shape: [1, seq_len, 4096]
            print(f"Activations shape: {activations.shape}")
        print(f"GPU VRAM Used during inference: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        return jsonify({"activations": activations.cpu().tolist()})
    except Exception as e:
        print(f"Error in gpu_forward: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    app.run(host="0.0.0.0", port=8081)