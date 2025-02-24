from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bitsandbytes import BitsAndBytesConfig
import psutil

app = Flask(__name__)

# Model setup
model_id = "SweatyCrayfish/llama-3-8b-quantized"
tokenizer = AutoTokenizer.from_pretrained(model_id)

try:
    print("Loading model with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    total_layers = len(model.model.layers)  # 32
    split_point = total_layers - 1  # 31, leave last layer for Mac
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
        position_ids = torch.tensor(data.get('position_ids', torch.arange(input_ids.shape[1]).unsqueeze(0).tolist())).to("cuda")
        print("Input_ids shape:", input_ids.shape)
        print("Attention_mask shape:", attention_mask.shape)
        print("Position_ids shape:", position_ids.shape)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[split_point + 1]  # After layer 31
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