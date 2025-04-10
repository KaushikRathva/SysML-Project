import torch
import torch.nn as nn
import torch.multiprocessing as mp

# Define a simple example module (e.g., an attention layer or any custom module)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

def worker(model, rank):
    # Each worker can now use the shared model
    print(f"Process {rank} model parameters:")
    for name, param in model.named_parameters():
        print(name, param.data)

if __name__ == '__main__':
    # Set the start method; "spawn" is the safest choice, especially on Windows and macOS
    mp.set_start_method("spawn", force=True)
    
    # Create the model and place its parameters in shared memory
    shared_model = MyModel()
    shared_model.share_memory()  # Moves model parameters to shared memory

    processes = []
    for rank in range(4):  # For example, spawn 4 processes
        p = mp.Process(target=worker, args=(shared_model, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
