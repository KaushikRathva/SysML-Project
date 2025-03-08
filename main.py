from ModelProfiling import *
from CPUPRofiling import *
from LayerMapping import *
from config import *


def main():
    cpuProfiling()
    modelProfiling()
    layerMapping()

if __name__ == "__main__":
    main()