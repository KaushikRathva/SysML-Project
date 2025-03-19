c2cLatency  = [ [0,  29, 28, 27],
                [29,  0, 28, 26],
                [28, 28,  0, 26],
                [27, 26, 26,  0]
                ]
k = 1024
m = k*k

L1_size =   320*k
L2_size =   5*m
LLC_size=   12*m

no_of_core = 4
LLC_per_core = LLC_size/no_of_core
Priv_Cache_per_core = L1_size + L2_size

corePrivCache   = [0 for i in range(no_of_core)]
coreSharedLLC   = [0 for i in range(no_of_core)]

for i in range(no_of_core):
    corePrivCache[i] = Priv_Cache_per_core
    coreSharedLLC[i] = LLC_per_core 

cpuProfile  = [c2cLatency, corePrivCache, coreSharedLLC]
print(cpuProfile)
# CPU profiling
def cpuProfiling():
    print("This is cpu profilling.")
    return cpuProfile
