# Import python file with model implementted.
import multiprocessing
import time
# from ModelProfiling import *
# from CPUProfiling import *
# import multiprocessing

def worker(send_conn, recv_conn, w_n):
    while True:
        message = recv_conn.recv()
        if message == 'exit':
            print(f'Worker {w_n} received exit signal')
            break
        print(f'Worker {w_n} received: {message}')
        # time.sleep(5)  # Wait for 5 seconds
        send_conn.send(f'Hello from worker {w_n}')
    recv_conn.close()
    send_conn.close()

class Deployer:
    pipes = {}
    layers = []
    # comp_blocks = []
    processes = []
    # def __init__(self, layers, blocks):
    def __init__(self, layers):
        self.layers = layers
        # self.comp_blocks = blocks
        # self.pipes = {layer: multiprocessing.Pipe() for layer in layers}
        self.pipes = {i: multiprocessing.Pipe() for i in range(len(layers) + 1)}
        print(self.pipes)
        # print(self.pipes)
    def create_processes(self,):
        len_layers = len(self.layers)
        for layer_idx in range(len_layers):
            p = multiprocessing.Process(target=self.layers[layer_idx], args=(self.pipes[layer_idx + 1][1], self.pipes[layer_idx][0], layer_idx))
            self.processes.append(p)
            p.start()
        try:
            while True:
                # Main process sends initial message to the first worker
                self.pipes[0][1].send('Hello from main process')

                # Main process receives final message from the last worker
                final_message = self.pipes[len_layers][0].recv()
                print(f'Main process received: {final_message}')
                print("#################################")
                
                time.sleep(5)  # Wait for 5 seconds
        except KeyboardInterrupt:
            print("Terminating processes...")
            for idx in range(len_layers):
                self.pipes[idx+1][1].send('exit')

        # Join all processes
        for p in self.processes:
            p.join()
    

layers = [worker, worker, worker, worker] 
deploy = Deployer(layers)

deploy.create_processes()