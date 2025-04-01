import torch
import torch.multiprocessing as mp
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Model Configuration ===
MODEL_NAME = "facebook/opt-1.3b"  # Change model as needed
NUM_CORES = torch.get_num_threads()  # Use all available CPU cores

# === Load Model & Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.share_memory()  # Allow multiple processes to access the model

# === Sample Input ===
input_text = "What is the future of AI?"
inputs = tokenizer(input_text, return_tensors="pt")

# === Function to Run Model on a Single Core ===
def run_inference(core_id, queue):
    torch.set_num_threads(1)  # Limit to a single thread per process
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    end_time = time.time()
    latency = end_time - start_time
    queue.put(latency)  # Store result in queue

# === Parallel Execution Across Cores ===
if __name__ == "__main__":
    try:
        mp.set_start_method("fork")  # Ensures correct multiprocessing behavior
    except RuntimeError:
        pass  # Ignore if context is already set

    queue = mp.Queue()  # ✅ Define queue before creating processes
    processes = []

    start_time = time.time()
    
    for core_id in range(NUM_CORES):
        p = mp.Process(target=run_inference, args=(core_id, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()

    # Collect latencies from all processes
    latencies = [queue.get() for _ in range(NUM_CORES)]
    
    avg_latency = sum(latencies) / len(latencies)
    throughput = NUM_CORES / (end_time - start_time)

    print(f"Average Latency per Core: {avg_latency:.4f} sec")
    print(f"Throughput across {NUM_CORES} cores: {throughput:.2f} requests per second")

    # === CPU Performance Counters (Using perf) ===
    #os.system(f"sudo perf stat -e cycles,instructions,cache-references,cache-misses taskset -c 0-{NUM_CORES-1} python -c 'import torch; torch.randn(1000,1000).mm(torch.randn(1000,1000))'")
    os.system(f"perf stat -e cycles,instructions,cache-references,cache-misses taskset -c 0-{NUM_CORES-1} python -c 'import torch; torch.randn(1000,1000).mm(torch.randn(1000,1000))'")

    # === Portability Check (System Info) ===
    #os.system("lscpu")
    #os.system("cat /proc/cpuinfo")
