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
    print(f"Loaded model with {len(model.model.layers)} layers")
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
            # Generate sequence
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=50,  # Adjust for desired length
                do_sample=True,     # Enable sampling for variety
                temperature=0.7,    # Control randomness
                top_p=0.95          # Nucleus sampling
            )
            print(f"Generated sequence shape: {outputs.shape}")
        
        print(f"GPU VRAM Used during inference: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        return jsonify({"generated_ids": outputs.cpu().tolist()})
    except Exception as e:
        print(f"Error in gpu_forward: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    app.run(host="0.0.0.0", port=8081)