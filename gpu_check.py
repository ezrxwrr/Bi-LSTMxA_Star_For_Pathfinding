import torch

print("=====================================")
# Mengecek apakah CUDA (GPU) tersedia
cuda_available = torch.cuda.is_available()
print(f"Apakah GPU tersedia untuk PyTorch? : {cuda_available}")

if cuda_available:
    # Mengambil nama GPU yang terdeteksi
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Nama GPU yang terdeteksi       : {gpu_name}")
    
    # Menampilkan total VRAM GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM GPU                 : {total_memory:.2f} GB")
else:
    print("GPU tidak terdeteksi. PyTorch masih menggunakan CPU.")
print("=====================================")