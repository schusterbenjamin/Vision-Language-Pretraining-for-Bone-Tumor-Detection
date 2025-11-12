import importlib

def check_package(package_name):
    try:
        importlib.import_module(package_name)
        print(f"[✓] {package_name} is installed.")
        return True
    except ImportError:
        print(f"[✗] {package_name} is NOT installed.")
        return False

def check_torch_cuda():
    try:
        import torch
        print(f"[✓] torch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[✓] CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[✗] CUDA is NOT available.")
    except ImportError:
        print("[✗] torch is NOT installed, cannot check CUDA support.")

def test_training():
    import torch
    from torch import nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Dummy model and data
    model = nn.Linear(10, 1).to(device)
    x = torch.randn(100, 10).to(device)
    y = torch.randn(100, 1).to(device)

    # Dummy training loop
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for i in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i+1}, Loss: {loss.item():.4f}")

def main():
    print("Checking package availability:\n")
    check_package("torch")
    check_package("lightning")
    check_package("wandb")
    check_package("hydra")
    
    print("\nChecking CUDA support for PyTorch:\n")
    check_torch_cuda()

    test_training()

if __name__ == "__main__":
    main()
