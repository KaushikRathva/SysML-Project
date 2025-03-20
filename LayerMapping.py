from ModelProfiling import modelProfiling
from CPUProfiling import cpuProfiling
import numpy as np

#All feasible core to block mappings
def exhaustive_search(cur_block_idx,occupied_cores,block_core_map,cores_per_block,total_blocks,cur_block_core_map):
    print("Current block id is ",cur_block_idx)
    print("Occupied cores so far are ",occupied_cores)
    print("Current Block core mapping is",cur_block_core_map)

    if(cur_block_idx == total_blocks):
        block_core_map.append(cur_block_core_map)
        occupied_cores={}
        return
    
    for cur_core in cores_per_block[cur_block_idx]:
        if cur_core not in occupied_cores:
            occupied_cores.add(cur_core)
            cur_block_core_map[cur_block_idx] = cur_core
            exhaustive_search(cur_block_idx+1,occupied_cores,block_core_map,cores_per_block,total_blocks,cur_block_core_map)

#Calculation of communication time after exhaustive search
#Input: List of dictionaries with keys = block id, value = core id, c2clatency 
def calc_communication(block_core_map,c2cLatency,modelTrafficProfile):
    for cur_map in block_core_map:
        for block1 in cur_map:
            for block2 in cur_map:
                core1 = cur_map[block1]
                core2 = cur_map[block2]
                comm_time = comm_time + modelTrafficProfile[block1][block2]*c2cLatency[core1][core2]
    return np.array(comm_time)

#Approximate modelTrafficProfile is highest between consecutive layers
#Input is c2cLatency and computeblocks from cpuprofiling
#Output is block to core mapping. Dictionary with keys = block index, value = core indec
def heuristic1(c2clatency,computeBlocks):
    c2clatency = np.array(c2clatency)
    block_core_map={}
    least_latency_cores = []
    least_latency_cores_idx = {}
    min_latency_per_core = np.argmin(c2clatency,axis = 1)
    second_min_latency_per_core = np.argmin(c2clatency[c2clatency != np.min(c2clatency,axis = 1)],axis = 1)
    num_blocks = len(computeBlocks)

    for start_idx in range(c2clatency.shape[0]):
        cur_idx = start_idx
        for idx in range(num_blocks):
            min_latency = min_latency_per_core[cur_idx]
            if min_latency in least_latency_cores_idx:
                min_latency = second_min_latency_per_core[cur_idx]
            cur_idx = min_latency
            least_latency_cores_idx.add(cur_idx)
            least_latency_cores.append(min_latency)
    
    for block_idx,_ in enumerate(computeBlocks):
        block_core_map[block_idx] = least_latency_cores[block_idx]
    return block_core_map

# Layer mapping
#Primary function: Input is search mode. Can be exhaustive or heuristic
#Output is leastcostmap. A function that provides best mapping from block to core. 
#Keys are block indices and values are core indices
def layerMapping(searchMode = "exhaustive"):
    
    #Get information of model profiling and CPU profiling
    modelProfile, computeBlockProfile = modelProfiling()
    Layers, modelTrafficProfile, modelMemProfile,memPerBlock = modelProfile
    computeBlocks, computeBlockDupliation = computeBlockProfile
    c2cLatency, corePrivCache, coreSharedLLC = cpuProfiling()

 
    #Constraint #1: If a single layer cannot be mapped onto one core(can be done in Phase 3)
    #Assumption: Single layer in a single core
    #Minimise communication while fitting memory requirements
    
    #Check cores that each layer can fit in
    cores_per_block = []
    corePrivCache = np.array(corePrivCache)
    coreSharedLLC = np.array(coreSharedLLC)
    total_core_mem = corePrivCache+coreSharedLLC
    for block_mem in memPerBlock:
        feasible_cores = (total_core_mem >= block_mem).nonzero()
        cores_per_block.append(feasible_cores)

    #Exhaustive Search Algorithm
    if(searchMode == "exhaustive"):
        block_core_map = []
        occupied_cores = {}
        st_block_idx = 0
        total_blocks = len(computeBlocks)
        cur_block_core_map = {}
        exhaustive_search(st_block_idx,occupied_cores,block_core_map,cores_per_block,total_blocks,cur_block_core_map)
        least_cost_map = block_core_map[np.argmin(calc_communication(block_core_map,modelTrafficProfile))]

    #Heuristic Search Algorithm
    if(searchMode == "heuristic"):
        least_cost_map = heuristic1(c2cLatency,computeBlocks)
        

        
    return least_cost_map


layerMapping()



