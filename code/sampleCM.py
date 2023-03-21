import numpy as np
import multiprocessing as mp
from multiprocessing.pool import Pool
from tqdm import tqdm

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

def sample_conf_model(N,degree_distrib):
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

def one_graph_job(N,degree_distrib):
    graph = sample_conf_model(N,degree_distrib)
    selfs = prop_self_edges(graph)
    mult = prop_multiple_edged_vertex(graph)
    return [selfs,mult]

def apply_pool(pool,N_sample,args):
    apply_async_futs =[]
    for _ in (range(N_sample)):
        apply_async_futs.append(pool.apply_async(one_graph_job,args))
    return [x.get() for x in tqdm(apply_async_futs)]

def main(path,N_sample,N,p,degree=[1,4],n_try=1000):
    proba  = [1-p,p]
    degree_distrib = get_degree_distrib(N,degree,proba,n_try)
    n_proc = int(input("How many cpu to use ? : "))
    with Pool(processes=n_proc) as pool :
        stats=apply_pool(pool,N_sample,(N,degree_distrib,))
    stats = np.array(stats)
    np.save(path,stats)


# x=np.array([4,4,4,4,4,4,4,4])
# N=x.shape[0]
# graph=sample_conf_model(N,x)
# print(graph.shape)
# vertex_counts = np.sum(graph, axis=1)
# multiple_edges = np.sum(np.equal(vertex_counts, 2))
# print(graph)
# print(vertex_counts,multiple_edges)



if __name__ == "__main__" :
    N_sample=1000
    N=2000
    p=.7
    expe = "2"
    path = "./experiment_"+expe+"/"
    file = f"N_sample{N_sample}_N{N}_p{p}"
    main(path+file,N_sample,N,p)
