import torch
import onnxruntime as ort

print("=== CUDA & cuDNN Check (PyTorch) ===")
if torch.cuda.is_available():
    print(f"✅ CUDA is available! Version: {torch.version.cuda}")
    print(f"💾 GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"📦 cuDNN version: {torch.backends.cudnn.version()}")
else:
    print("❌ CUDA is NOT available in PyTorch!")

print("\n=== ONNX Runtime GPU Check ===")
try:
    providers = ort.get_available_providers()
    print("Available providers:", providers)
    if "CUDAExecutionProvider" in providers:
        print("✅ ONNX Runtime can use GPU (CUDAExecutionProvider is available).")
    else:
        print("⚠️ GPU support NOT available for ONNX Runtime. Falling back to CPU.")
except Exception as e:
    print(f"❌ Failed to query ONNX Runtime providers: {e}")

print("\nTest complete.")
