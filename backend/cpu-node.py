from flask import Flask, request, Response, jsonify
import torch
from transformers import AutoTokenizer
import psutil
import time
from flask_cors import CORS
import gc
import grpc
import llm_pb2
import llm_pb2_grpc

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])

# Use a model that doesn't require sentencepiece
model_id = "google/gemma-2-9b-it"  # This uses GPT-2 tokenizer which doesn't need sentencepiece
tokenizer = AutoTokenizer.from_pretrained(model_id)

# GPU node configuration
GPU_NODE_ADDRESS = "10.0.0.196:50051"  # Replace with your Windows GPU node's IP and gRPC port

@app.route('/generate', methods=['POST'])
def generate_cover_letter():
    def generate(user_message=None):
        if user_message:
            # For DialoGPT, we'll use a simpler prompt format
            prompt = f"User: {user_message.strip()}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs['input_ids'][0].tolist()
            attention_mask = inputs['attention_mask'][0].tolist()

            # Send input_ids and attention_mask to GPU node
            try:
                channel = grpc.insecure_channel(GPU_NODE_ADDRESS)
                stub = llm_pb2_grpc.LLMServiceStub(channel)
                request = llm_pb2.ForwardRequest(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                response = stub.Forward(request, timeout=60)
                if response.generated_ids:
                    result = tokenizer.decode(response.generated_ids, skip_special_tokens=True)
                else:
                    result = "[GPU node returned no tokens]"
            except Exception as e:
                print(f"Error contacting GPU node: {e}")
                result = "[Error: GPU node unavailable]"

            for word in result.split():
                yield f"data: {word} \n\n"
                time.sleep(0.05)
        yield "data: \n\n"

    if request.method == 'POST':
        data = request.get_json()
        user_message = data.get('job_description', '') if data else ''
        return Response(generate(user_message), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/unload_model', methods=['POST'])
def unload_model():
    global tokenizer
    tokenizer = None
    gc.collect()
    print("Tokenizer unloaded from memory.")
    return jsonify({"status": "unloaded"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)