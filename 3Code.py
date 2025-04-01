import torch
import torch.nn as nn
import torch.multiprocessing as mp
import time
import os

# Define a simple Transformer model (similar to the decoder part of GPT)
class SimpleTransformer(nn.Module):
    def __init__(self, num_tokens, embedding_dim, nhead, num_layers, dim_feedforward):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embedding_dim, nhead, dim_feedforward),
            num_layers
        )
        self.fc = nn.Linear(embedding_dim, num_tokens)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, memory):
        src = self.embedding(src)
        output = self.transformer_decoder(src, memory)
        output = self.fc(output)
        return output

# Function to run the model on a specific core
def run_model_on_core(model, input_data, output_queue):
    core_id = os.getpid()  # Get process ID which can be loosely associated with a core
    device = torch.device("cpu")
    model.to(device)
    input_tensor = input_data.to(device)
    memory = torch.randn(input_tensor.size(0), 1, model.embedding.embedding_dim).to(device) # Dummy memory

    print(f"Process {core_id}: Started processing input of size {input_tensor.shape}")

    start_time = time.time()
    output = model(input_tensor, memory)
    end_time = time.time()

    latency = end_time - start_time
    throughput = input_tensor.size(0) / latency if latency > 0 else 0

    # Placeholder for FLOPS measurement (requires external tools)
    flops = "Measurement not implemented here"

    # Placeholder for Energy Consumption measurement (requires system-level monitoring)
    energy_consumption = "Measurement not implemented here"

    output_queue.put({
        "core_id": core_id,
        "latency": latency,
        "throughput": throughput,
        "flops": flops,
        "energy_consumption": energy_consumption
    })

    print(f"Process {core_id}: Finished processing in {latency:.4f} seconds.")

if __name__ == "__main__":
    # **Significantly Reduced Model Parameters (for testing)**
    num_tokens = 5000  # Reduced
    embedding_dim = 64   # Reduced
    nhead = 1      # Reduced
    num_layers = 1   # Reduced
    dim_feedforward = 128 # Reduced

    # **Significantly Reduced Input Data (for testing)**
    batch_size = 16    # Reduced
    seq_len = 16     # Reduced
    input_data = torch.randint(0, num_tokens, (seq_len, batch_size))

    # Number of cores to use
    num_cores = mp.cpu_count()
    # num_cores = 1
    print(f"Number of available CPU cores: {num_cores}")

    # Divide the input data
    input_chunks = torch.split(input_data, batch_size // num_cores if batch_size >= num_cores else 1, dim=1)
    num_processes = len(input_chunks)
    print(f"Number of processes created: {num_processes}")

    # Create the model
    model = SimpleTransformer(num_tokens, embedding_dim, nhead, num_layers, dim_feedforward)
    model.share_memory()
    print("Model initialized and weights shared.")

    output_queue = mp.Queue()
    processes = []

    start_time_total = time.time()
    print("Starting parallel processing...")

    for i in range(num_processes):
        print(f"Starting process {i+1} with input shape: {input_chunks[i].shape}")
        process = mp.Process(target=run_model_on_core, args=(model, input_chunks[i], output_queue))
        processes.append(process)
        process.start()

    # Collect results from each process
    print("Waiting for all processes to finish...")
    all_results = []
    for _ in range(num_processes):
        try:
            result = output_queue.get(timeout=300)  # Added a timeout of 5 minutes
            print(f"Received result from process {result['core_id']}")
            all_results.append(result)
        except mp.queues.Empty:
            print("Timeout: One or more processes did not finish within the allocated time.")
            break

    for process in processes:
        process.join(timeout=60) # Added a timeout for joining processes
        if process.is_alive():
            print(f"Warning: Process {process.pid} did not terminate properly.")
            process.terminate()

    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    total_throughput = sum(result['throughput'] for result in all_results) if all_results else 0

    print("\n--- Performance Results ---")
    print(f"Total execution time: {total_time:.4f} seconds")
    print(f"Total throughput: {total_throughput:.2f} samples/second")

    if all_results:
        for result in all_results:
            print(f"\nCore/Process ID: {result['core_id']}")
            print(f"  Latency: {result['latency']:.4f} seconds")
            print(f"  Throughput: {result['throughput']:.2f} samples/second")
            print(f"  FLOPS: {result['flops']}")
            print(f"  Energy Consumption: {result['energy_consumption']}")
    else:
        print("No results were collected from the processes.")

    print("\n--- Portability ---")
    print("This implementation uses PyTorch and the multiprocessing module, which are portable across different operating systems and hardware that support Python and PyTorch.")