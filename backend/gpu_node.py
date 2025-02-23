from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Model setup
model_id = "SweatyCrayfish/llama-3-8b-quantized"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load first half of the model (layers 0-15)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
total_layers = len(model.model.layers)  # Should be 32 for LLaMA 3 8B
split_point = total_layers // 2  # 16
gpu_layers = torch.nn.Sequential(*model.model.layers[:split_point]).to("cuda")
print(f"Loaded {split_point} layers on GPU")

@app.route('/gpu_forward', methods=['POST'])
def gpu_forward():
    data = request.get_json()
    input_ids = torch.tensor(data['input_ids']).to("cuda")
    with torch.no_grad():
        activations = gpu_layers(input_ids)
    print(f"GPU VRAM Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    return jsonify({"activations": activations.cpu().tolist()})

if __name__ == "__main__":
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    app.run(host="0.0.0.0", port=8081)  # Runs on Windows