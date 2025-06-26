import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import grpc
from concurrent import futures
import llm_pb2
import llm_pb2_grpc

class LLMServiceServicer(llm_pb2_grpc.LLMServiceServicer):
    def __init__(self):
        self.model_id = "microsoft/phi-2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        print("Loading model on GPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            torch_dtype=torch.float16,
            device_map="cuda:0"
        )
        print(f"Loaded model with {len(self.model.model.layers)} layers")
        print(f"VRAM Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"RAM Available: {psutil.virtual_memory().available / 1024**3:.2f} GB")

    def Forward(self, request, context):
        try:
            input_ids = torch.tensor([request.input_ids], dtype=torch.long).to("cuda")
            attention_mask = torch.tensor([request.attention_mask], dtype=torch.long).to("cuda")
            position_ids = torch.tensor([request.position_ids], dtype=torch.long).to("cuda")
            print("Input_ids shape:", input_ids.shape)
            print("Attention_mask shape:", attention_mask.shape)
            print("Position_ids shape:", position_ids.shape)

            with torch.no_grad():
                # Forward pass through all layers (or partial, if splitting)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    return_dict=True
                )
                # Return the last hidden state as activations
                activations = outputs.hidden_states[-1].detach().cpu().flatten().tolist()

            print(f"Returning activations of length: {len(activations)}")
            return llm_pb2.ForwardResponse(activations=activations, message="Success")
        except Exception as e:
            print(f"Error in Forward: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return llm_pb2.ForwardResponse(activations=[], message=str(e))

    def HealthCheck(self, request, context):
        return llm_pb2.HealthResponse(status="healthy")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    llm_pb2_grpc.add_LLMServiceServicer_to_server(LLMServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC server running on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()