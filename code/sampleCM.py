import numpy as np
import multiprocessing as mp
from multiprocessing.pool import Pool
from tqdm import tqdm
from scipy.sparse import coo_array
import pickle

def prop_self_edges(graph):
    # M = |{Edges}|
    M = graph.shape[0]
    return sum([graph[m,0]==graph[m,1] for m in range(M)])/M

def prop_multiple_edged_vertex(graph):
    M = graph.shape[0]
    set = len({(i,j) for [i,j] in np.sort(graph)})
    return (M-set)/M

def get_degree_distrib(N,degree,proba,n_try):
    for n in range(n_try):
        distrib = np.random.choice(degree,N,p=proba)
        if distrib.sum()%2 == 0 :
            return distrib
    return None

def sample_conf_model(N,degree,proba,n_try):
    degree_distrib = get_degree_distrib(N,degree,proba,n_try)
    M = degree_distrib.sum()//2
    stubs = np.array([ j  for j in range(N) for i in range(degree_distrib[j])])
    p_stubs = np.array([1 for i in range(2*M)]) 
    p_stubs = p_stubs/p_stubs.sum() 
    graph = []
    for _ in range(M):
        pair = np.random.choice(stubs,size=2,replace=False,p=p_stubs/p_stubs.sum())
        graph.append(pair)
        for x in pair :
            k = np.argwhere(stubs==x).reshape(-1)[0]
            stubs[k]=-1
            p_stubs[k]=0
    return np.array(graph)

def get_stats_self_mult(N,degree_distrib):
    graph = sample_conf_model(N,degree_distrib)
    selfs = prop_self_edges(graph)
    mult = prop_multiple_edged_vertex(graph)
    return [selfs,mult]

def one_graph_job(N,degree,proba,n_try):
    graph = sample_conf_model(N,degree,proba,n_try)
    matrix = coo_array((np.ones(graph.shape[0]),tuple(graph.T)), shape=(N,N))
    return matrix

def apply_pool(pool,N_sample,args):
    apply_async_futs =[]
    for _ in (range(N_sample)):
        apply_async_futs.append(pool.apply_async(one_graph_job,args))
    return [x.get() for x in (apply_async_futs)]

def main(path,n_proc,N_sample,N,p,degree=[1,4],n_try=1000):
    proba  = [1-p,p]
    #n_proc = int(input("How many cpu to use ? : "))
    with Pool(processes=n_proc) as pool :
        results=apply_pool(pool,N_sample,(N,degree,proba,n_try,))
    np.save(path,np.array(results))



if __name__ == "__main__" :
    n_proc=11
    N_sample=10
    N=[100,1000,10000]
    Proba=np.linspace(0,1,500)
    path = "./graphs/"
    for n in N :
        for p in tqdm(Proba) :
            file = f"N_sample{N_sample}_N{n}_p{p:.4f}"
            main(path+file,n_proc,N_sample,n,p)
