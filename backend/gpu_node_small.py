import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import psutil
import grpc
from concurrent import futures
import llm_pb2
import llm_pb2_grpc
import os

class LLMServiceServicer(llm_pb2_grpc.LLMServiceServicer):
    def __init__(self):
        try:
            # Use a smaller model for testing
            self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Much smaller than Mistral-7B
            print("Loading model config and tokenizer on GPU node...")
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Please install CUDA-enabled PyTorch.")
            
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"VRAM Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"RAM Available: {psutil.virtual_memory().available / 1024**3:.2f} GB")
            
            # Load tokenizer and config
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.config = AutoConfig.from_pretrained(self.model_id)
            
            # TinyLlama has 22 layers, let's load layers 11-21 (last half)
            layer_start = 11
            layer_end = 22
            
            # Create a custom device map that only loads specific layers to GPU
            device_map = {}
            device_map["model.embed_tokens"] = "cpu"  # Keep embedding on CPU
            device_map["model.norm"] = "cpu"  # Keep norm on CPU
            device_map["lm_head"] = "cuda:0"  # Keep lm_head on GPU
            
            # Map only layers 11-21 to GPU
            for i in range(layer_start, layer_end):
                device_map[f"model.layers.{i}"] = "cuda:0"
            
            # Map all other layers to CPU (they won't be loaded)
            for i in range(22):  # TinyLlama has 22 layers
                if i < layer_start or i >= layer_end:
                    device_map[f"model.layers.{i}"] = "cpu"
            
            print(f"Loading layers {layer_start} to {layer_end-1} on GPU...")
            
            # Load model with custom device map
            full_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=device_map,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Extract only the layers we need
            self.layers = torch.nn.ModuleList()
            for i in range(layer_start, layer_end):
                self.layers.append(full_model.model.layers[i])
            
            self.lm_head = full_model.lm_head
            self.layer_start = layer_start
            self.layer_end = layer_end
            
            print(f"Successfully loaded layers {layer_start} to {layer_end-1} on GPU")
            print(f"VRAM Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"RAM Available: {psutil.virtual_memory().available / 1024**3:.2f} GB")
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error in __init__: {e}")
            raise

    def Forward(self, request, context):
        try:
            # Reconstruct activations tensor
            batch = request.batch
            seq_len = request.seq_len
            hidden_dim = request.hidden_dim
            activations = torch.tensor(request.activations, dtype=torch.float16).reshape(batch, seq_len, hidden_dim).to("cuda")
            attention_mask = torch.tensor(request.attention_mask, dtype=torch.long).reshape(batch, seq_len).to("cuda")
            position_ids = torch.tensor(request.position_ids, dtype=torch.long).reshape(batch, seq_len).to("cuda")
            print("Received activations shape:", activations.shape)
            print("Attention_mask shape:", attention_mask.shape)
            print("Position_ids shape:", position_ids.shape)

            with torch.no_grad():
                hidden_states = activations
                for i, layer in enumerate(self.layers):
                    print(f"Processing layer {self.layer_start + i}")
                    layer_output = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        output_attentions=False
                    )
                    hidden_states = layer_output[0]
                
                logits = self.lm_head(hidden_states)

            print(f"Returning logits of shape: {logits.shape}")
            return llm_pb2.ForwardResponse(
                logits=logits.detach().cpu().flatten().tolist(),
                batch=batch,
                seq_len=seq_len,
                vocab_size=logits.shape[-1],
                message="Success"
            )
        except Exception as e:
            print(f"Error in Forward: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return llm_pb2.ForwardResponse(logits=[], batch=0, seq_len=0, vocab_size=0, message=str(e))

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