import numpy as np
import matplotlib.pyplot as plt

N=[100,200,300,400,500,600,700,800,900,1000,2000]
n=6
f1="proportion_multiple_edged_vertex_p0."
f2="proportion_self_edge_p0."



selfprop=[]
multprop=[]
for i in range(1,n+1):
    x=np.load("./experiment_1/"+f1 +str(i)+".npy")
    selfprop.append(x)
    
    x=np.load("./experiment_1/"+f2 +str(i)+".npy")
    multprop.append(x)
selfprop=np.array(selfprop).mean(axis=-1)
multprop=np.array(multprop).mean(axis=-1)

print(selfprop.shape,multprop.shape)
for i  in range(n):
    plt.plot(N,selfprop[i],label='selfprop p=0.'+str(i+1))
plt.legend()
plt.show()

for i  in range(n):
    plt.plot(N,multprop[i],label='multprop p=0.'+str(i+1))
plt.legend()
plt.show()



file="N_sample10_N2000_p0.7"

data_2 = np.load("./experiment_2/N_sample10_N2000_p0.7.npy")
print(data_2.shape)
selfprop=data_2[:,0]
multprop=data_2[:,1]
print(selfprop.mean()),multprop.mean()
plt.plot(selfprop)
plt.show
plt.plot(multprop)
plt.show()

# plt.plot(selfprop.T[10])
# plt.plot(multprop.T[10])
# plt.show()

# for i in range(n):
#     for j in range(len(N)):
#         plt.hist(selfprop[i,j],alpha=.2,label="selfprop "+str(i)+str("\t")+str(N[j]))
#         plt.legend()
#         plt.show()