from flask import Flask, request, Response, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import time
import requests

app = Flask(__name__)

# Model setup
model_id = "SweatyCrayfish/llama-3-8b-quantized"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
total_layers = len(model.model.layers)  # 32
split_point = total_layers // 2  # 16
cpu_layers = torch.nn.Sequential(*model.model.layers[split_point:]).to("cpu")
lm_head = model.lm_head.to("cpu")
print(f"Loaded {total_layers - split_point} layers on CPU")

GPU_NODE_URL = "http://localhost:8081/gpu_forward"  # Adjust if Windows is on a different IP

print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")

@app.route('/generate', methods=['POST'])
def generate_cover_letter():
    data = request.get_json()
    job_description = data.get('job_description', '') if data else ''
    
    prompt = f"{job_description}"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']

    # Step 1: Offload to Windows GPU
    response = requests.post(GPU_NODE_URL, json={"input_ids": input_ids.tolist()})
    response_json = response.json()
    print("Windows response keys:", response_json.keys())  # Debug keys
    
    if 'error' in response_json:
        return jsonify({"error": f"Windows GPU node failed: {response_json['error']}"}), 500
    if 'activations' not in response_json:
        return jsonify({"error": "No 'activations' key in Windows response"}), 500
    
    gpu_activations = torch.tensor(response_json['activations']).to("cpu")
    print("GPU activations shape:", gpu_activations.shape)

    # Step 2: Process on Mac CPU
    with torch.no_grad():
        cpu_outputs = cpu_layers(gpu_activations)
        logits = lm_head(cpu_outputs)

    generated_ids = torch.argmax(logits, dim=-1)
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return jsonify({"generated_text": result})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)