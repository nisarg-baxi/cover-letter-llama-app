clearimport torch
import psutil

def test_cuda():
    print("=== CUDA Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        device_props = torch.cuda.get_device_properties(0)
        print(f"GPU name: {device_props.name}")
        print(f"GPU memory: {device_props.total_memory / 1024**3:.2f} GB")
        print(f"VRAM used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"VRAM cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        print("CUDA is not available!")
        print("You need to install CUDA-enabled PyTorch.")
        print("Try: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print(f"\n=== System Info ===")
    print(f"RAM available: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    print(f"RAM total: {psutil.virtual_memory().total / 1024**3:.2f} GB")

if __name__ == "__main__":
    test_cuda() 