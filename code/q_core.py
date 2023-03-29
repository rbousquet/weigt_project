import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from color import shiftedColorMap 

def velocities(p1,p2,p3,p4):
    p_inf = p1+p2
    p_sup = p3+p4 
    
    Vp4 = - p4*(1-p_sup**4)
    Vp3 = - p3 * (1-p_sup**3) + 4*p4 * p_inf * p_sup**3  
    Vp2 = - p2 + 6*p4 * p_inf**2 * p_sup**2 + 3*p3*p_inf*p_inf**2
    Vp1 = - p1 + 4*p4 * p_inf**3 * p_sup + 3*p3 * p_inf**2* p_sup + 2*p2 * p_inf * p_sup
    Vp0 =   p1 + p2 + p3 * p_inf**3 + p4* p_inf**4

    return Vp0,Vp1,Vp2,Vp3,Vp4

def trajectories(p04,max_iter,dt=1e-4):
    p4 = p04
    p3 = 0
    p2 = 0
    p1 = 1-p4
    p0 = 0
    X = []
    V = []
    for _ in range(max_iter) :
        X.append([p0,p1,p2,p3,p4])
        Vp0,Vp1,Vp2,Vp3,Vp4=velocities(p1,p2,p3,p4)
        V.append([Vp0,Vp1,Vp2,Vp3,Vp4])
        p0 += Vp0*dt
        p1 += Vp1*dt
        p2 += Vp2*dt
        p3 += Vp3*dt
        p4 += Vp4*dt

        s = sum([p1,p2,p3,p4])
        p1/=s
        p2/=s
        p3/=s
        p4/=s
        
    return [X,V]

MAX=1000
N_spaces=100
p03 = 0
p04 = np.linspace(0,1,N_spaces)
S =[]
for p4 in p04:
    S.append(trajectories(p4,max_iter=MAX,dt=1e-2))
S=np.array(S)

#viridis = shiftedColorMap(mpl.colormaps['coolwarm'],start=0, midpoint=0.8, stop=1.0)
viridis = mpl.colormaps['coolwarm']
magma = mpl.colormaps['magma']#resampled(100)
fig,ax = plt.subplots(1,2,gridspec_kw={'width_ratios': [5, 1]})
ax[0].set_xlabel("time")
ax[0].set_ylabel("3-core size (%) ")
for i in range(N_spaces):
    ax[0].plot(  (S[i,0,:,3]+S[i,0,:,4]) ,c=viridis(p04[i]))
    #plt.plot(S.sum(axis=-1)[i,0,:],c=magma(1-p04[i]))
mpl.colorbar.ColorbarBase(ax[1], cmap=viridis)
ax[1].set_xlabel(r"$\pi$ ")
#plt.yscale("log")
plt.tight_layout()
plt.savefig("./figures/3_core_size_vs_time.png")
plt.show()



print(S.shape)

plt.plot(p04,S[:,0,-1,3]+S[:,0,-1,4])
#plt.plot(np.linspace(0,1,100),[1/9 for i in range(100)],)
plt.xlabel(r" $\pi$ ")
plt.ylabel("3-core final size (%)")
plt.legend()
plt.savefig("./figures/3_core_final_size_vs_pi.png")
plt.show()


Vmax = np.max(S[:,1,:,3]+S[:,1,:,4],axis=-1)
print(Vmax.shape)
plt.plot(p04,Vmax)
plt.xlabel(r" $\pi$ ")
plt.ylabel("3-core max velocity")
plt.legend()
plt.savefig("./figures/3_core_Vmax_vs_pi.png")
plt.show()

