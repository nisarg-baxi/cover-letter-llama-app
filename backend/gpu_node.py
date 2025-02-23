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
    total_layers = len(model.model.layers)  # Should be 32
    split_point = total_layers // 2  # 16
    print(f"Loaded model with {total_layers} layers, split at {split_point}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print(f"VRAM Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"RAM Available: {psutil.virtual_memory().available / 1024**3:.2f} GB")

@app.route('/gpu_forward', methods=['POST'])
def gpu_forward():
    try:
        data = request.get_json()
        input_ids = torch.tensor(data['input_ids']).to("cuda")
        attention_mask = torch.tensor(data.get('attention_mask', [[1] * input_ids.shape[1]])).to("cuda")
        print("Input_ids shape:", input_ids.shape)
        print("Attention_mask shape:", attention_mask.shape)
        
        with torch.no_grad():
            # Forward pass up to split_point
            hidden_states = model.model.embed_tokens(input_ids)  # Shape: [1, seq_len, 4096]
            print("Embeddings shape:", hidden_states.shape)
            for i in range(split_point):
                hidden_states = model.model.layers[i](hidden_states, attention_mask=attention_mask)[0]
            print(f"Activations shape after {split_point} layers: {hidden_states.shape}")
        
        print(f"GPU VRAM Used during inference: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        return jsonify({"activations": hidden_states.cpu().tolist()})
    except Exception as e:
        print(f"Error in gpu_forward: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    app.run(host="0.0.0.0", port=8081)