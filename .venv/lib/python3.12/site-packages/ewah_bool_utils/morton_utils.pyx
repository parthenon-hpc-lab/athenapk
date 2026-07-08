import numpy as np

cimport cython
cimport numpy as np


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.uint32_t morton_neighbors_coarse(np.uint64_t mi1, np.uint64_t max_index1,
                                         bint periodicity[3], np.uint32_t nn,
                                         np.uint32_t[:,:] index,
                                         np.uint64_t[:,:] ind1_n,
                                         np.uint64_t[:] neighbors):
    cdef np.uint32_t ntot = 0
    cdef np.uint64_t ind1[3]
    cdef np.uint32_t count[3]
    cdef np.uint32_t origin[3]
    cdef np.int64_t adv
    cdef int i, j, k, ii, ij, ik
    for i in range(3):
        count[i] = 0
        origin[i] = 0
    # Get indices
    decode_morton_64bit(mi1,ind1)
    # Determine which directions are valid
    for j,i in enumerate(range(-nn,(nn+1))):
        if i == 0:
            for k in range(3):
                ind1_n[j,k] = ind1[k]
                index[count[k],k] = j
                origin[k] = count[k]
                count[k] += 1
        else:
            for k in range(3):
                adv = <np.int64_t>((<np.int64_t>ind1[k]) + i)
                if (adv < 0):
                    if periodicity[k]:
                        while adv < 0:
                            adv += max_index1
                        ind1_n[j,k] = <np.uint64_t>(adv % max_index1)
                    else:
                        continue
                elif (adv >= max_index1):
                    if periodicity[k]:
                        ind1_n[j,k] = <np.uint64_t>(adv % max_index1)
                    else:
                        continue
                else:
                    ind1_n[j,k] = <np.uint64_t>(adv)
                # print(i,k,adv,max_index1,ind1_n[j,k],adv % max_index1)
                index[count[k],k] = j
                count[k] += 1
    # Iterate over ever combinations
    for ii in range(count[0]):
        i = index[ii,0]
        for ij in range(count[1]):
            j = index[ij,1]
            for ik in range(count[2]):
                k = index[ik,2]
                if (ii != origin[0]) or (ij != origin[1]) or (ik != origin[2]):
                    neighbors[ntot] = encode_morton_64bit(ind1_n[i,0],
                                                          ind1_n[j,1],
                                                          ind1_n[k,2])
                    ntot += 1
    return ntot

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.uint32_t morton_neighbors_refined(np.uint64_t mi1, np.uint64_t mi2,
                                          np.uint64_t max_index1, np.uint64_t max_index2,
                                          bint periodicity[3], np.uint32_t nn,
                                          np.uint32_t[:,:] index,
                                          np.uint64_t[:,:] ind1_n,
                                          np.uint64_t[:,:] ind2_n,
                                          np.uint64_t[:] neighbors1,
                                          np.uint64_t[:] neighbors2):
    cdef np.uint32_t ntot = 0
    cdef np.uint64_t ind1[3]
    cdef np.uint64_t ind2[3]
    cdef np.uint32_t count[3]
    cdef np.uint32_t origin[3]
    cdef np.int64_t adv, maj, rem, adv1
    cdef int i, j, k, ii, ij, ik
    for i in range(3):
        count[i] = 0
        origin[i] = 0
    # Get indices
    decode_morton_64bit(mi1,ind1)
    decode_morton_64bit(mi2,ind2)
    # Determine which directions are valid
    for j,i in enumerate(range(-nn,(nn+1))):
        if i == 0:
            for k in range(3):
                ind1_n[j,k] = ind1[k]
                ind2_n[j,k] = ind2[k]
                index[count[k],k] = j
                origin[k] = count[k]
                count[k] += 1
        else:
            for k in range(3):
                adv = <np.int64_t>(ind2[k] + i)
                maj = adv // (<np.int64_t>max_index2)
                rem = adv % (<np.int64_t>max_index2)
                if adv < 0:
                    adv1 = <np.int64_t>(ind1[k] + (maj-1))
                    if adv1 < 0:
                        if periodicity[k]:
                            while adv1 < 0:
                                adv1 += max_index1
                            ind1_n[j,k] = <np.uint64_t>adv1
                        else:
                            continue
                    else:
                        ind1_n[j,k] = <np.uint64_t>adv1
                    while adv < 0:
                        adv += max_index2
                    ind2_n[j,k] = <np.uint64_t>adv
                elif adv >= max_index2:
                    adv1 = <np.int64_t>(ind1[k] + maj)
                    if adv1 >= max_index1:
                        if periodicity[k]:
                            ind1_n[j,k] = <np.uint64_t>(adv1 % <np.int64_t>max_index1)
                        else:
                            continue
                    else:
                        ind1_n[j,k] = <np.uint64_t>adv1
                    ind2_n[j,k] = <np.uint64_t>rem
                else:
                    ind1_n[j,k] = ind1[k]
                    ind2_n[j,k] = <np.uint64_t>(adv)
                index[count[k],k] = j
                count[k] += 1
    # Iterate over ever combinations
    for ii in range(count[0]):
        i = index[ii,0]
        for ij in range(count[1]):
            j = index[ij,1]
            for ik in range(count[2]):
                k = index[ik,2]
                if (ii != origin[0]) or (ij != origin[1]) or (ik != origin[2]):
                    neighbors1[ntot] = encode_morton_64bit(ind1_n[i,0],
                                                           ind1_n[j,1],
                                                           ind1_n[k,2])
                    neighbors2[ntot] = encode_morton_64bit(ind2_n[i,0],
                                                           ind2_n[j,1],
                                                           ind2_n[k,2])
                    ntot += 1
    return ntot
