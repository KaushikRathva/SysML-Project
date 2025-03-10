d_model = 512
seq_len = 512
p = 4

E = d_model*seq_len
h = 1
dk = d_model/h
dv = dk

Wq = d_model * dk
Wk = d_model * dk
Wv = d_model * dv

Wff1 = d_model*4*d_model
Wff2 = 4*d_model*d_model

comuteBlocks = ["MHA", "Add&Norm", "FFN", "Linear", "Softmax"]
#TO                         MHA     A&N     FFN     Linear Softmax 
modelTrafficProfile = [ [       0,      E,      0,      0,      0],#MHA from
                        [       0,      E,      E,      E,      0],#A&N
                        [       0,      E,      0,      0,      0],#FFN
                        [       0,      0,      0,      0,      E],#Linear
                        [       E,      0,      0,      0,      0]]#SoftMAX


modelMemProfile     = [h*(Wq+Wk+Wv+3*E),0,Wff1+Wff2+E+2,E+2,  E+1]
for i in range(len(modelMemProfile)):
    modelMemProfile[i] *= p

# Model profilling 
def modelProfiling():
    print("This is model profiling")