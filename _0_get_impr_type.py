
import numpy as np
import copy

def get_impr_type(affinity_networks,thres_type):
    type_num=len(affinity_networks)
    patient_num=affinity_networks[0].shape[0]
    ls_adjs=[]
    type_adj = np.zeros((patient_num, patient_num))  # 4*4

    init_type_flag = 1
    for i in range(type_num):
        ls_adj_i=[]
        affinity_networks_else = copy.deepcopy(affinity_networks)
        del affinity_networks_else[i]
        for j in range(type_num-1):
            ls_adj_i.append((affinity_networks[i]-affinity_networks_else[j])/affinity_networks_else[j]>thres_type)  # 4*4
        ls_adj_i=np.array(ls_adj_i)
        sum_ls_adj_i=ls_adj_i.sum(axis=0)

        type_adj=np.where(sum_ls_adj_i==type_num-1,init_type_flag*np.ones((patient_num,patient_num)),type_adj)
        init_type_flag+=1

        ls_adjs.append(ls_adj_i)

    affinity_networks_sorted=np.sort(np.array(affinity_networks),axis=0)
    def is_10perc_smaller(max_nda, min_nda):
        return (max_nda - min_nda) / max_nda < thres_type
    condition=is_10perc_smaller(affinity_networks_sorted[1],affinity_networks_sorted[0])\
    &is_10perc_smaller(affinity_networks_sorted[2],affinity_networks_sorted[0])\
    &is_10perc_smaller(affinity_networks_sorted[2],affinity_networks_sorted[1])

    type_adj = np.where(condition, 7 * np.ones((patient_num, patient_num)), type_adj)

    affinity_networks_nda=np.array(affinity_networks)
    max_bool = np.where(affinity_networks_nda==affinity_networks_nda.max(axis=0), True, False )
    min_bool = np.where(affinity_networks_nda==affinity_networks_nda.min(axis=0), True, False)

    type_adj = np.where(min_bool[2] & np.where(type_adj==0,True,False), 4 * np.ones((patient_num, patient_num)), type_adj)
    type_adj = np.where(min_bool[1] & np.where(type_adj==0,True,False), 5 * np.ones((patient_num, patient_num)), type_adj)
    type_adj = np.where(min_bool[0] & np.where(type_adj==0,True,False), 6 * np.ones((patient_num, patient_num)), type_adj)

    return type_adj
