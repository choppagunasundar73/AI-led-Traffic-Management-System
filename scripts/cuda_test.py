import torch
import torchvision

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print("-" * 30)

if not torch.cuda.is_available():
    print("❌ CUDA is not available. PyTorch was not installed with GPU support.")
else:
    print(f"✅ CUDA is available!")
    print(f"CUDA Version (built-in): {torch.version.cuda}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Create dummy data and move it to the GPU
        boxes = torch.rand(10, 4).to('cuda') * 100
        scores = torch.rand(10).to('cuda')
        
        # This is the exact operation that fails in your tracker
        print("\nAttempting torchvision.ops.nms on GPU...")
        indices = torchvision.ops.nms(boxes, scores, 0.5)
        
        print(f"✅ SUCCESS: 'torchvision::nms' ran successfully on the GPU.")
        print(f"Resulting indices: {indices}")

    except Exception as e:
        print("\n❌ FAILED: The 'torchvision::nms' operation failed on the GPU.")
        print("This confirms the installation issue.")
        print("\nError Details:")
        print(e)