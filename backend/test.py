import requests

sample_input = {
    "input_ids": [[1, 2, 3, 4, 5]],  # Replace with actual token IDs
    "attention_mask": [[1, 1, 1, 1, 1]]
}

response = requests.post("http://10.0.0.196:8081/gpu_forward", json=sample_input)
print(response.json())