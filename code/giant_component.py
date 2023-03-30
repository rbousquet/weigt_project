import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from scipy.sparse import coo_array
from multiprocessing.pool import Pool



def get_dataset(path,N_sample,N,P) :
    data=[]
    for n in N :
        proba=[]
        print(f"Importing data N={n} ...")
        for p in P :
            file = path+f"/N_sample{N_sample}_N{n}_p{p:.4f}.npy"
            proba.append(np.load(file,allow_pickle=True))
        data.append(proba)
    return np.array(data)

def biggest_component_size(N,graph) :
    """
    graph : adjacency matrix (here a coo_array)
    """
    G = np.argwhere(graph == 1)
    clusters=[set(G[0])]

    for edge in G :
        e = set(edge)
        for c in clusters :
            if bool(c & e) :
                c.update(e)
                break
        clusters.append(e)

    for x in clusters :
        for y in clusters[1:] :
            if len(x.intersection(y)) != 0 :
                x.update(y)
                clusters.remove(y)
    return np.max([len(c) for c in clusters])/N

def apply_pool(pool,N,graphs):
    apply_async_futs =[]
    for graph in graphs:
        apply_async_futs.append(pool.apply_async(biggest_component_size,(N,graph,)))
    return [x.get() for x in (apply_async_futs)]

def main(path,n_proc,N,graphs):
    with Pool(processes=n_proc) as pool :
        results=apply_pool(pool,N,graphs)
    np.save(path,np.array(results))

if __name__ == "__main__" :
    available = mp.cpu_count()
    n_proc = int(input(f"How many cpu to use ? ({available} available) : "))
    path_in = "./graphs"
    path = "./giant_component/"
    N_sample=10
    N = np.array([100,1000])
    P = np.linspace(0,1,500)
    data = get_dataset(path_in,N_sample,N,P)
    for n in range(N.size) :
        print(f"Computing component size for N={N[n]} ...")
        for p in tqdm(range(P.size)) :
            file = f"N_sample{N_sample}_N{N[n]}_p{P[p]:.4f}"
            main(path+file,n_proc,N[n],data[n,p])

    