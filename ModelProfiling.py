d_model = 512
seq_len = 512

E = d_model*seq_len

comuteBlocks = ["MHA", "Add&Norm", "FFN", "Linear", "Softmax"]
#                 MHA  A&N FFN Linear", "Softmax"
modelProfile = [[  0,  E,  ],
                [],
                [],
                [],
                []]

# Model profilling 
def modelProfiling():
    print("This is model profiling")