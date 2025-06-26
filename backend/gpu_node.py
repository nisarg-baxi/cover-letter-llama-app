import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import grpc
from concurrent import futures
import llm_pb2
import llm_pb2_grpc

class LLMServiceServicer(llm_pb2_grpc.LLMServiceServicer):
    def __init__(self):
        self.model_id = "mistralai/Mistral-7B-v0.1"
        print("Loading model config and tokenizer on GPU node...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        full_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="cuda:0"
        )
        # Only keep the last 16 layers and lm_head
        total_layers = len(full_model.model.layers)
        split_point = total_layers // 2
        self.layers = torch.nn.ModuleList(full_model.model.layers[split_point:]).to("cuda")
        self.lm_head = full_model.lm_head.to("cuda")
        self.config = full_model.config
        print(f"Loaded layers {split_point} to {total_layers-1} on GPU")
        print(f"VRAM Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"RAM Available: {psutil.virtual_memory().available / 1024**3:.2f} GB")

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
                for layer in self.layers:
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