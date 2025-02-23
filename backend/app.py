from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])
print("Loading model from ... mistralai/Mistral-7B-v0.1")
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
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