# Import python file with model implementted.
import multiprocessing
from ModelProfiling import *
# import multiprocessing

class Deployer:
    def __init__(self, layers):
        self.pipes = {}
        self.layers = layers
        self.processes = []
        for layer in layers:
            self.pipes[layer] =multiprocessing.Pipe()
    def create_processes(self,):
        prev_layers = []
        for layer in self.layers:
            p = multiprocessing.Process(target=layer, args=(self.pipes[layer], self.pipes[prev_layers[0]]))
            prev_layers = [layer]
            self.processes.append(p)
            p.start()
            