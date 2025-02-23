from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import psutil
import time
import os

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173", "http://10.0.0.239:5173"])

# Check resources
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")

# Model setup for Llama 3 8B Quantized
model_id = "SweatyCrayfish/llama-3-8b-quantized"  # Updated model ID

print(f"Loading model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
)
print("Model loaded")

@app.route('/generate', methods=['POST', 'GET'])
def generate_cover_letter():
    def generate(job_description=None):
        if job_description:
            # Tailored prompt for cover letter
            prompt = f"{job_description}"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

            # Generate with low token limit for speed/memory
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,  # Fits your memory, adjustable
                do_sample=True,
                temperature=0.6,  # Recommended by DeepSeek for coherence
                top_p=0.95
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Stream words
            for word in result.split():
                yield f"data: {word} \n\n"
                time.sleep(0.05)
        yield "data: \n\n"

    if request.method == 'POST':
        data = request.get_json()
        job_description = data.get('job_description', '')
        return Response(generate(job_description), mimetype='text/event-stream')

    elif request.method == 'GET':
        return Response(generate(), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)