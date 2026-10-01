import os
import numpy as np
from numba.typed import List
from numba import njit
from pathlib import Path
import re
import matplotlib.pyplot as plt

cheetah3_x_size = 512
color_list = ['lightblue', 'skyblue','deepskyblue','dodgerblue','slateblue',
              'royalblue','mediumblue','blue','darkslateblue','navy']

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

def find_clusters(matrix: np.ndarray):
    num_vars = matrix.shape[0]
    # Fix 1: Use bool instead of np.bool
    visited = np.zeros((num_vars,)).astype(bool)
    clusters = []

    for node in range(num_vars):
        if not visited[node]:
            # Start a new cluster tracking list
            current_cluster = []
            queue = [node]
            visited[node] = True
            while queue:
                u = queue.pop(0)
                current_cluster.append(u)
                # Check all potential neighbors in the matrix
                for v in range(num_vars):
                    if matrix[u][v] == 1 and not visited[v]:
                        visited[v] = True
                        queue.append(v)

            clusters.append(current_cluster)
    return clusters

@njit
def build_adj_mat(xs : np.ndarray, ys : np.ndarray) : 
    num_vars = xs.shape[0]
    xmat = np.outer(xs,np.ones((num_vars,),dtype=xs.dtype))
    ymat = np.outer(ys,np.ones((num_vars,),dtype=xs.dtype))
    diff_xmat = np.abs(xmat - xmat.T)
    diff_ymat = np.abs(ymat -ymat.T)
    diff_tot = diff_xmat + diff_ymat
    mat = (np.logical_and(diff_tot > 0, diff_tot < 5)).astype(xs.dtype)
    return mat

@njit
def clusterize(data : np.ndarray, time_span : int = 300, max_num_clust : int = 5) :
    # print('In clusterize')
    data = data.astype(np.int64)
    len_data = data.shape[1]
    ref_ind = 0
    glob_i = 0
    cluster_results = np.ones((3, max_num_clust, len_data//2),dtype=data.dtype)*-1
    while ref_ind < len_data :
        ind_step = ref_ind + 1
        if ind_step >= len_data :
            return cluster_results
        while data[0,ind_step] - data[0,ref_ind] < time_span :
            ind_step += 1
            if ind_step >= len_data :
                return cluster_results
        xs = data[2,ref_ind:ind_step]
        ys = data[3,ref_ind:ind_step]
        toas = data[0,ref_ind:ind_step]
        ref_ind = ind_step
        adj_mat = build_adj_mat(xs,ys)
        chunk_clusters = find_clusters_numba(adj_mat, max_num_clust)
        cluster_inds = np.where(chunk_clusters >=0)
        cls_count = np.bincount(cluster_inds[0])
        temp_cluster_res = np.ones((3,max_num_clust),dtype=data.dtype)*-1
        for pos, cls in enumerate(cluster_inds[0]) :
            temp_cluster_res[0,cls] += xs[cluster_inds[1][pos]]
            temp_cluster_res[1,cls] += ys[cluster_inds[1][pos]]
            temp_cluster_res[2,cls] += toas[cluster_inds[1][pos]]
        for i in np.arange(cls_count.shape[0]) :
            temp_cluster_res[0,i] //= cls_count[i]
            temp_cluster_res[1,i] //= cls_count[i]
            temp_cluster_res[2,i] //= cls_count[i]
        cluster_results[:,:,glob_i] = temp_cluster_res
        glob_i +=1
    return cluster_results

@njit
def clusterize_single_chunk(data : np.ndarray, max_num_clust) :
    data = data.astype(np.int64)
    xs = data[2,:]
    ys = data[3,:]
    adj_mat = build_adj_mat(xs,ys)
    chunk_clusters = find_clusters_numba(adj_mat, max_num_clust)
    cluster_inds = np.where(chunk_clusters >=0)
    cls_count = np.bincount(cluster_inds[0])
    temp_cluster_res = np.ones((2,max_num_clust),dtype=data.dtype)*-1
    for pos, cls in enumerate(cluster_inds[0]) :
        temp_cluster_res[0,cls] += xs[cluster_inds[1][pos]]
        temp_cluster_res[1,cls] += ys[cluster_inds[1][pos]]
    for i in np.arange(cls_count.shape[0]) :
        temp_cluster_res[0,i] //= cls_count[i]
        temp_cluster_res[1,i] //= cls_count[i]
    return temp_cluster_res

def single_chunk_cluster(data) :
    xs = data[2,:]
    ys = data[3,:]
    adj_mat = build_adj_mat(xs,ys)
    chunk_clusters = find_clusters(adj_mat)
    cluster_pos = []
    for clust in chunk_clusters :
        x_mean = np.mean(xs[clust])
        y_mean = np.mean(ys[clust])
        cluster_pos.append([x_mean,y_mean])
    return cluster_pos

def clusterize_files(input_folder_path : str,
                     output_folder_path : str,
                     time_span : int = 300,
                     max_num_clust : int = 5,
                     output_order = 3) -> None :
    file_list = os.listdir(input_folder_path)
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir()
    for file in file_list :
        if file.endswith('.npy') :
            file_path = input_folder_path / Path(file)
            sorted_array = np.load(file_path)
            clustered_array = clusterize(sorted_array,time_span=time_span,max_num_clust=max_num_clust)
            x_arr = clustered_array[0]
            nb_valides = (x_arr != -1).sum(axis=0)
            for i in range(1,output_order+1) :
                filename = f"{file[:-4]}_{i}th.npy"
                output_file_path = output_folder_path / filename
                indices = np.where(nb_valides == i)[0]
                if indices.size > 0 :
                    nth_array = x_arr[:i, indices]
                    np.save(output_file_path,nth_array)

def find_zlp_pos(input_singlets) :
    return np.mean(input_singlets)

def calibrate_data(input_data : np.ndarray, zlp_pos : float, energy_dispersion : float) :
    centered_data = input_data - zlp_pos
    calibrated_data = centered_data*energy_dispersion
    energy_range = [-zlp_pos*energy_dispersion,(cheetah3_x_size-zlp_pos)*energy_dispersion]
    return calibrated_data, energy_range

def sort_multiplets(input_data : np.ndarray) :
    input_data.sort(axis=0)
    return input_data

def produce_numpy_histogram(input_data,
                            energy_range : list[float,float],
                            order : int) :
    output_tuple = None
    match order :
        case 1 :
            output_tuple = [np.histogram(input_data,
                                        bins=cheetah3_x_size,
                                        range=energy_range)]
        # case 2 :
        #     output_tuple = np.histogram2d(input_data[0],
        #                                   input_data[1],
        #                                   bins=[cheetah3_x_size,cheetah3_x_size],
        #                                   range=[energy_range,energy_range])
        case _ :
            # image = np.zeros((512,512))
            output_tuple = []
            for i in range(order) :
                output = np.histogram(input_data[i],
                                      bins=cheetah3_x_size,
                                      range=energy_range)
                output_tuple.append(output)
            # input_data = input_data.reshape(input_data.shape[0],input_data.shape[1]*input_data.shape[2])
                # im, xedge, yedge = np.histogram2d(input_data[i-1,:],input_data[i,:],
                #                                     bins=[cheetah3_x_size,cheetah3_x_size],
                #                                     range=[energy_range,energy_range])
                # image += im
            # output_tuple = (image,xedge,yedge)
    return output_tuple

def produce_nth_histogram(input_tuple : np.ndarray,
                          dim : int) :
    fig, ax = plt.subplots(figsize=(8, 5))
    for i,input in enumerate(input_tuple) :
        ax.stairs(input[0],
                    input[1],
                    fill=True,
                    alpha=0.5,
                    color=color_list[i%len(color_list)],
                    edgecolor='black',
                    label=f'{i}-th electron')
        ax.set_xlabel("Energy loss (eV)", fontsize=11)
        ax.set_ylabel("Counts", fontsize=11)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)

    return fig,ax

def prepare_sum_histo(input_folder_path : str) :
    files = os.listdir(input_folder_path)
    numbers = set()
    for file in files :
        if file.endswith('.npy') :
            file_path = input_folder_path / Path(file)
            nth_match = nth_match = re.match(r"(.*)([0-9])(th)(.*)",file)
            if nth_match :
                number = nth_match.group(2)
                numbers.add(number)
    number_list = list(numbers)
    number_dict = {}
    for num in number_list :
        tuple_list = [
            [np.zeros((cheetah3_x_size,)), np.zeros((cheetah3_x_size + 1,))]
            for _ in range(int(num))
            ]
        number_dict[num] = tuple_list
    return number_dict

def produce_histograms(input_folder_path : str,
                       output_folder_path : str,
                       energy_dispersion : float,
                       zlp_pos = None
                       ) :
    files = os.listdir(input_folder_path)
    number_dict = prepare_sum_histo(input_folder_path)
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir()
    for file in files :
        if file.endswith('.npy') :
            file_path = input_folder_path / Path(file)
            nth_match = re.match(r"(.*)([0-9])(th)(.*)",file)
            if nth_match :
                number = nth_match.group(2)
                nuplets = np.load(file_path)
                match number :
                    case "1" :
                        if zlp_pos is None :
                            zlp_pos = find_zlp_pos(nuplets)
                    case _ :
                        nuplets = sort_multiplets(nuplets)
                assert zlp_pos is not None, "Something is wrong with the zlp position"
                # cal_data, energy_range = calibrate_data(nuplets,
                #                                         zlp_pos,
                #                                         energy_dispersion)
                # histogram_tuple = produce_numpy_histogram(cal_data,
                #                                           energy_range,
                #                                           order = int(number))
                histogram_tuple = produce_numpy_histogram(nuplets,
                                                          [0,cheetah3_x_size],
                                                          order = int(number))
                for i, tuple in enumerate(histogram_tuple) :
                    number_dict[number][i][0] += tuple[0]
                    number_dict[number][i][1] = tuple[1]
                fig,ax = produce_nth_histogram(histogram_tuple,
                                               int(number))
                ax.set_title(file)
                plt.tight_layout()
                output_filepath = output_folder_path / Path(f"{file[:-4]}_singlet_histo.png")
                fig.savefig(output_filepath, dpi=300, bbox_inches="tight")
                plt.close()
    for key,item in number_dict.items() :
        fig,ax = produce_nth_histogram(item,int(key))
        output_filepath = output_folder_path / Path(f"sum_{key}th_histo.png")
        fig.savefig(output_filepath, dpi=300, bbox_inches="tight")
        plt.close()
                        
    

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