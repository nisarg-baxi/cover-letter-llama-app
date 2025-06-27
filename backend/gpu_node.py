import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import psutil
import grpc
from concurrent import futures
import llm_pb2
import llm_pb2_grpc
import os

# Disable Triton to avoid installation issues
os.environ["DISABLE_TRITON"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

class LLMServiceServicer(llm_pb2_grpc.LLMServiceServicer):
    def __init__(self):
        self.model_id = "microsoft/DialoGPT-medium"  # Use the same model as CPU node
        print("Loading tokenizer and 4-bit quantized model on GPU node...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,  # Disable double quantization to avoid Triton
            bnb_4bit_quant_type="nf4"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cuda:0",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True
        )
        print(f"Loaded full model on GPU in 4-bit mode")
        print(f"VRAM Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"RAM Available: {psutil.virtual_memory().available / 1024**3:.2f} GB")

    def Forward(self, request, context):
        try:
            input_ids = torch.tensor(request.input_ids, dtype=torch.long).unsqueeze(0).to("cuda")
            attention_mask = torch.tensor(request.attention_mask, dtype=torch.long).unsqueeze(0).to("cuda")
            print("Input_ids shape:", input_ids.shape)
            print("Attention_mask shape:", attention_mask.shape)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=40,
                    repetition_penalty=1.2
                )
                generated_ids = outputs[0].detach().cpu().tolist()

            print(f"Returning generated_ids of length: {len(generated_ids)}")
            return llm_pb2.ForwardResponse(
                generated_ids=generated_ids,
                message="Success"
            )
        except Exception as e:
            print(f"Error in Forward: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return llm_pb2.ForwardResponse(generated_ids=[], message=str(e))

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