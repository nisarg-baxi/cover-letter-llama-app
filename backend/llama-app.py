from flask import Flask, request, Response, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import time

app = Flask(__name__)

# Model setup
model_id = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
eos_token_id = tokenizer.eos_token_id

try:
    print("Loading Gemma-2-2B-IT in FP16 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    print(f"Loaded model with {len(model.model.layers)} layers in FP16")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print(f"EOS token ID: {eos_token_id}")
print(f"RAM Available: {psutil.virtual_memory().available / 1024**3:.2f} GB")

@app.route('/generate', methods=['POST'])
def generate_cover_letter():
    start_time = time.time()
    data = request.get_json()
    job_description = data.get('job_description', '') if data else ''
    
    # Instruction-tuned prompt
    prompt = f"<|prompt|>Human: {job_description}\nAssistant: "
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    input_ids = inputs['input_ids'][:1]  # Shape: [1, seq_length]
    attention_mask = inputs['attention_mask'][:1]  # Shape: [1, seq_length]
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to("cpu")  # Shape: [1, seq_length]

    print("Input_ids shape:", input_ids.shape)
    print("Attention_mask shape:", attention_mask.shape)
    print("Position_ids shape:", position_ids.shape)
    print("Input text:", tokenizer.decode(input_ids[0], skip_special_tokens=True))
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.2,
            cache_implementation="sliding_window",
            eos_token_id=None  # Prevent early stopping
        )
        print(f"Generated sequence shape: {outputs.shape}")
        print(f"Generated tokens: {outputs[0].tolist()}")
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated text:", result)

    print(f"Generation time: {time.time() - start_time:.2f} seconds")
    print(f"RAM Used: {psutil.virtual_memory().used / 1024**3:.2f} GB")
    return jsonify({"generated_text": result})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)