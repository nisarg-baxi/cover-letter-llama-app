from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print(f"Total RAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else "No GPU")
import psutil
print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")
import time
import os

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])
print("Loading model from ... mistralai/Mistral-7B-v0.1")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("CUDA not detected by PyTorch")
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Loading tokenizer..")
model = AutoModelForCausalLM.from_pretrained(model_id)
print("NM: Model loaded")

if torch.cuda.is_available():
    model = model.to('cuda') # Use GPU if available

@app.route('/generate', methods=['POST', 'GET'])
def generate_cover_letter():
    def generate(job_description=None):
        if job_description:
            prompt = f"{job_description}"
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')

            outputs = model.generate(**inputs, max_new_tokens=200)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            for word in result.split():
                yield f"data: {word} \n\n"
                time.sleep(0.1)
        yield "data: \n\n"

    if request.method == 'POST':
        data = request.get_json()
        job_description = data['job_description']
        return Response(generate(job_description), mimetype='text/event-stream')
    
    elif request.method == 'GET':
        return Response(generate(), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=8080)