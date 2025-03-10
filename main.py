from ModelProfiling import *
from CPUProfiling import *
from LayerMapping import *
from config import *

# keep all the reconfigurable parameters in config.py

def main():
    cpuProfiling()
    modelProfiling()
    layerMapping()

if __name__ == "__main__":
    main()