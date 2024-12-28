import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

if __name__ == "__main__":
    check_cuda()