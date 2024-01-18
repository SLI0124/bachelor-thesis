import torch


def main():
    print(torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")


if __name__ == "__main__":
    main()
