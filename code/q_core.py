import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def velocities(p3,p4):
    Vp4 = -p4*(1-(p3+p4)**4)
    Vp3 = -p3*(1-(p3+p4)**3)+4*p4*( (p3+p4)**3 - (p3+p4)**4 ) 
    Vp0 = -Vp3 -Vp4
    return Vp0,Vp3,Vp4

def trajectories(p03,p04,max_iter,dt=1e-4):
    p3,p4=p03,p04
    p0 = 1 - (p3+p4)
    X = []
    V = []
    for _ in range(max_iter) :
        X.append([p0,p3,p4])
        Vp0,Vp3,Vp4=velocities(p3,p4)
        V.append([Vp0,Vp3,Vp4])
        p0 += Vp0*dt
        p3 += Vp3*dt
        p4 += Vp4*dt
    return [X,V]

MAX=1000
p03 = 0
p04 = np.linspace(0,1,100)
#x = np.array(trajectories(p03,p04[50],max_iter=1000,dt=1e-2))
S =[]
for p4 in p04:
    S.append(trajectories(p03,p4,max_iter=MAX,dt=1e-2))
S=np.array(S)

viridis = matplotlib.colormaps['viridis']#resampled(100)
magma = matplotlib.colormaps['magma']#resampled(100)
for i in range(100):
    plt.plot(S[i,0,:,1]+S[i,0,:,2],c=viridis(1-p04[i]))
    plt.plot(S[i,0,:,0],c=magma(1-p04[i]))

plt.xlabel("time")
plt.ylabel("3 core proportion")
#plt.yscale("log")
plt.show()




#for i in range(0,MAX,MAX//10) :
plt.plot(p04,S[:,0,-1,0]+S[:,0,-1,1],label=f"time={i}")
#plt.plot(np.linspace(0,1,100),[1/9 for i in range(100)],)
plt.xlabel(r"p4 initial")
plt.ylabel("final size")
plt.legend()
plt.show()



for i in range(100):
    plt.plot(S[i,1,:,0]+S[i,1,:,1],c=viridis(1-p04[i]))
plt.xlabel("time")
plt.ylabel("3-core velocity")
plt.show()

