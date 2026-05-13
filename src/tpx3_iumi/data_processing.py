import numpy as np
from numba.typed import List
from numba import njit

def sort_data(sorting_col_index : int,data: np.ndarray) -> np.ndarray :
    """
    Sorts 2D data according to the n-th entry.

    Parameters
    ----------
    sorting_col_index : int
        n-th column that is used for sorting the data, e.g. time
    data : np.ndarray
        2D numpy data. First axis is expected to be categories of data and 2nd axis is expected to be occurences.

    Results
    -------
    sorted_data : np.ndarray
        sorted data according to the sorting_col_index column
    """
    sorting_inds = np.argsort(data[sorting_col_index])
    return data[:,sorting_inds]

def save_sorted_data(output_filename : str, input_filename : str, sorting_col_index : int) -> np.ndarray :
    """
    Saves sorted data.
    See sort_data for details.
    """
    data_to_sort = np.load(input_filename)
    np.save(output_filename,sort_data(sorting_col_index,data_to_sort))


@njit
def find_clusters_numba(matrix : np.ndarray, max_num_clust : int = 5):
    num_vars = matrix.shape[0]
    visited = np.zeros((num_vars,)).astype(np.bool)
    clusters = np.ones((max_num_clust,num_vars),dtype=matrix.dtype)*-1

    i_clust = -1
    for node in range(num_vars):
        if not visited[node]:
            # Start a new cluster
            queue = List([node])
            visited[node] = True
            i_clust+=1
            
            while queue:
                u = queue.pop(0)
                clusters[i_clust,u] = u
                
                # Check all potential neighbors in the matrix
                for v in range(num_vars):
                    if matrix[u][v] == 1 and not visited[v]:
                        visited[v] = True
                        queue.append(v)
            
            # clusters.append(current_cluster)
            
    return clusters

@njit
def build_adj_mat(xs : np.ndarray, ys : np.ndarray) : 
    num_vars = xs.shape[0]
    xmat = np.outer(xs,np.ones((num_vars,),dtype=xs.dtype))
    ymat = np.outer(ys,np.ones((num_vars,),dtype=xs.dtype))
    diff_xmat = np.abs(xmat - xmat.T)
    diff_ymat = np.abs(ymat -ymat.T)
    diff_tot = diff_xmat + diff_ymat
    mat = (diff_tot == 1).astype(xs.dtype)
    return mat

@njit
def clusterize(data : np.ndarray, time_span : int = 300, max_num_clust : int = 5) : 
    # print('In clusterize')
    data = data.astype(np.int64)
    len_data = data.shape[1]
    ref_ind = 0
    glob_i = 0
    cluster_results = np.ones((2, max_num_clust, len_data//2),dtype=data.dtype)*-1
    while ref_ind < len_data :
        ind_step = ref_ind + 1
        if ind_step >= len_data : 
            return cluster_results
        while data[0,ind_step] - data[0,ref_ind] < time_span :
            ind_step += 1
            if ind_step >= len_data : 
                return cluster_results
        # print(f"ind_step {ind_step}")
        # print(f"ref_ind : {ref_ind}")
        xs = data[2,ref_ind:ind_step]
        ys = data[3,ref_ind:ind_step]
        # print(f"xs : {xs.shape}")
        ref_ind = ind_step
        # print(f"ref ind : {ref_ind}")
        # print(f"ind step : {ind_step}")
        # print('adj')
        adj_mat = build_adj_mat(xs,ys)
        # print('find')
        chunk_clusters = find_clusters_numba(adj_mat, max_num_clust)
        cluster_inds = np.where(chunk_clusters >=0)
        temp_cluster_res = np.ones((2,max_num_clust),dtype=data.dtype)*-1
        for pos, cls in enumerate(cluster_inds[0]) : 
            temp_cluster_res[0,cls] += xs[cluster_inds[1][pos]]
            temp_cluster_res[1,cls] += ys[cluster_inds[1][pos]]
        temp_cluster_res[0] //= xs.shape[0]
        temp_cluster_res[1] //= ys.shape[0]
        cluster_results[:,:,glob_i] = temp_cluster_res
        glob_i +=1
    return cluster_results

if __name__ == '__main__' : 
    data  = np.load("sorted_apr27_17h.npy")
    import time
    start = time.time()
    clusters = clusterize(data[:,:10050])
    end = time.time()
    print(f"10050 entries  : {end-start}")
    start = time.time()
    clusters = clusterize(data[:,:100050])
    end = time.time()
    print(f"100050 entries  : {end-start}")
    start = time.time()
    clusters = clusterize(data[:,:1000050])
    end = time.time()
    print(f"1000050 entries  : {end-start}")
    start = time.time()
    clusters = clusterize(data[:,:10000050], max_num_clust=10)
    end = time.time()
    print(f"10000050 entries  : {end-start}")
    start = time.time()
    clusters = clusterize(data[:,:100000050],max_num_clust=10)
    end = time.time()
    print(f"100000050 entries  : {end-start}")