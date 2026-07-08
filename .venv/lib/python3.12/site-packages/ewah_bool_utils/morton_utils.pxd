"""
Helper functions to generate Morton indices




"""

cimport cython
cimport numpy as np

cdef extern from *:
    """
    const int XSHIFT=2;
    const int YSHIFT=1;
    const int ZSHIFT=0;
    """
    cdef int XSHIFT, YSHIFT, ZSHIFT

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.uint64_t compact_64bits_by2(np.uint64_t x):
    # Reversed magic
    x=x&(<np.uint64_t>0x1249249249249249)
    x=(x|(x>>2))&(<np.uint64_t>0x0649249249249249)
    x=(x|(x>>2))&(<np.uint64_t>0x0199219243248649)
    x=(x|(x>>2))&(<np.uint64_t>0x00786070C0E181C3)
    x=(x|(x>>4))&(<np.uint64_t>0x0007E007C00F801F)
    x=(x|(x>>10))&(<np.uint64_t>0x000001FFC00003FF)
    x=(x|(x>>20))&(<np.uint64_t>0x00000000001FFFFF)
    return x

#-----------------------------------------------------------------------------
# 21 bits spread over 64 with 2 bits in between
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.uint64_t spread_64bits_by2(np.uint64_t x):
    # This magic comes from http://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
    # Only reversible up to 2097151
    # Select highest 21 bits (Required to be reversible to 21st bit)
    # x = ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---k jihg fedc ba98 7654 3210
    x=(x&(<np.uint64_t>0x00000000001FFFFF))
    # x = ---- ---- ---- ---- ---- ---k jihg fedc ba-- ---- ---- ---- ---- --98 7654 3210
    x=(x|(x<<20))&(<np.uint64_t>0x000001FFC00003FF)
    # x = ---- ---- ---- -kji hgf- ---- ---- -edc ba-- ---- ---- 9876 5--- ---- ---4 3210
    x=(x|(x<<10))&(<np.uint64_t>0x0007E007C00F801F)
    # x = ---- ---- -kji h--- -gf- ---- -edc ---- ba-- ---- 987- ---6 5--- ---4 32-- --10
    x=(x|(x<<4))&(<np.uint64_t>0x00786070C0E181C3)
    # x = ---- ---k ji-- h--g --f- ---e d--c --b- -a-- --98 --7- -6-- 5--- -43- -2-- 1--0
    x=(x|(x<<2))&(<np.uint64_t>0x0199219243248649)
    # x = ---- -kj- -i-- h--g --f- -e-- d--c --b- -a-- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x=(x|(x<<2))&(<np.uint64_t>0x0649249249249249)
    # x = ---k --j- -i-- h--g --f- -e-- d--c --b- -a-- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x=(x|(x<<2))&(<np.uint64_t>0x1249249249249249)
    return x

@cython.cdivision(True)
cdef inline np.uint64_t encode_morton_64bit(np.uint64_t x_ind, np.uint64_t y_ind, np.uint64_t z_ind):
    cdef np.uint64_t mi
    mi = 0
    mi |= spread_64bits_by2(z_ind)<<ZSHIFT
    mi |= spread_64bits_by2(y_ind)<<YSHIFT
    mi |= spread_64bits_by2(x_ind)<<XSHIFT
    return mi

@cython.cdivision(True)
cdef inline void decode_morton_64bit(np.uint64_t mi, np.uint64_t *p):
    p[0] = compact_64bits_by2(mi>>XSHIFT)
    p[1] = compact_64bits_by2(mi>>YSHIFT)
    p[2] = compact_64bits_by2(mi>>ZSHIFT)

cdef np.uint32_t morton_neighbors_coarse(np.uint64_t mi1, np.uint64_t max_index1,
                                         bint periodicity[3], np.uint32_t nn,
                                         np.uint32_t[:,:] index,
                                         np.uint64_t[:,:] ind1_n,
                                         np.uint64_t[:] neighbors)

cdef np.uint32_t morton_neighbors_refined(np.uint64_t mi1, np.uint64_t mi2,
                                          np.uint64_t max_index1, np.uint64_t max_index2,
                                          bint periodicity[3], np.uint32_t nn,
                                          np.uint32_t[:,:] index,
                                          np.uint64_t[:,:] ind1_n,
                                          np.uint64_t[:,:] ind2_n,
                                          np.uint64_t[:] neighbors1,
                                          np.uint64_t[:] neighbors2)
