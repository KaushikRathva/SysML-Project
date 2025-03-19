import multiprocessing as mp
import time
import numpy as np
from multiprocessing import shared_memory, Barrier

def ping_pong(shm_name, size, barrier, sender_core, receiver_core):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    buffer = np.ndarray((size,), dtype=np.uint8, buffer=existing_shm.buf)
    
    # Pin process to a specific core if supported
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([sender_core])
    except (ImportError, AttributeError):
        pass
    
    barrier.wait()  # Synchronize before starting
    
    start_time = time.perf_counter()
    buffer[0] = 1  # Write to shared memory
    
    timeout = time.perf_counter() + 1  # 1-second timeout to prevent deadlock
    while buffer[0] == 1:
        if time.perf_counter() > timeout:
            print(f"Timeout: Core {sender_core} -> Core {receiver_core}")
            break
    
    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1e6  # Convert to microseconds
    print(f"Latency from Core {sender_core} to Core {receiver_core}: {latency:.2f} µs")
    
    existing_shm.close()

def receiver(shm_name, size, receiver_core):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    buffer = np.ndarray((size,), dtype=np.uint8, buffer=existing_shm.buf)
    
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([receiver_core])
    except (ImportError, AttributeError):
        pass
    
    while buffer[0] == 0:
        pass  # Wait for sender
    
    buffer[0] = 0  # Acknowledge reception
    existing_shm.close()

def measure_latency(sender_core, receiver_core):
    size = 1  # Smallest message size
    shm = shared_memory.SharedMemory(create=True, size=size)
    buffer = np.ndarray((size,), dtype=np.uint8, buffer=shm.buf)
    buffer[0] = 0  # Initialize memory
    
    barrier = Barrier(2)  # Synchronization barrier
    
    p1 = mp.Process(target=ping_pong, args=(shm.name, size, barrier, sender_core, receiver_core))
    p2 = mp.Process(target=receiver, args=(shm.name, size, receiver_core))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    
    shm.close()
    shm.unlink()

if __name__ == "__main__":
    core_pairs = [(0, 1), (0, 2), (1, 2)]  # Adjust based on available cores
    for sender, receiver in core_pairs:
        measure_latency(sender, receiver)
