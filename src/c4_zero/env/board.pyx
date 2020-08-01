import numpy as np
cimport numpy as np

ctypedef np.npy_intp SIZE_t  # Type for indices and counters
ctypedef np.npy_float64 DOUBLE_t  # Type of y, sample_weight
ctypedef unsigned long long ull

DEF BH = 6
DEF BW = 7
DEF MOVES = 42
DEF A = 4
DEF BLACK = 1
DEF WHITE = -1
DEF EMPTY = 0

cpdef int popcnt(ull n):
    cdef ull c = 0
    c = (n & 0x5555555555555555) + ((n >> 1) & 0x5555555555555555)
    c = (c & 0x3333333333333333) + ((c >> 2) & 0x3333333333333333)
    c = (c & 0x0f0f0f0f0f0f0f0f) + ((c >> 4) & 0x0f0f0f0f0f0f0f0f)
    c = (c & 0x00ff00ff00ff00ff) + ((c >> 8) & 0x00ff00ff00ff00ff)
    c = (c & 0x0000ffff0000ffff) + ((c >> 16) & 0x0000ffff0000ffff)
    c = (c & 0x00000000ffffffff) + ((c >> 32) & 0x00000000ffffffff)
    return <int> c

cpdef Board create_board_from_array2d(np.ndarray[SIZE_t, ndim=2] cell_ary):
    cdef int i, y, x
    cdef Board ret
    cdef ull black, white
    cdef ull b = 1
    black, white = 0, 0
    for y in range(BH):
        for x in range(BW):
            i = y * BW + x
            if cell_ary[y, x] == BLACK:
                black |= (b << i)
            if cell_ary[y, x] == WHITE:
                white |= (b << i)

    ret = Board()
    ret.black = black
    ret.white = white
    return ret

cdef class Board:
    # 64マスまで利用可能
    cdef public ull black
    cdef public ull white

    def __init__(self):
        self.black = 0
        self.white = 0

    cpdef Board copy(self):
        ret = Board()
        ret.black = self.black
        ret.white = self.white
        return ret

    cpdef Board reversed_copy(self):
        ret = Board()
        ret.black = self.white
        ret.white = self.black
        return ret

    cpdef int get_value(self, int y, int x):
        cdef int i
        cdef ull b = 1
        i = y * BW + x
        if (self.black & (b << i)) > 0:
            return BLACK
        if (self.white & (b << i)) > 0:
            return WHITE
        return EMPTY

    cpdef void set_value(self, int y, int x, int v):
        cdef int i
        cdef ull b = 1
        i = y * BW + x
        if v == BLACK:
            self.black |= (b << i)
        if v == WHITE:
            self.white |= (b << i)

    cpdef np.ndarray[SIZE_t, ndim=1] to_array1d(self):
        cdef int i
        cdef np.ndarray[SIZE_t, ndim=1] ary
        cdef ull b = 1
        ary = np.zeros(MOVES, dtype=int)
        for i in range(MOVES):
            if (self.black & (b << i)) > 0:
                ary[i] = BLACK
            if (self.white & (b << i)) > 0:
                ary[i] = WHITE
        return ary

    cpdef np.ndarray[SIZE_t, ndim=1] legal_moves(self):
        cdef int i
        cdef ull b = 1
        ret = np.zeros(MOVES, dtype=int)
        for x in range(BW):
            for y in reversed(range(BH)):
                i = y * BW + x
                if ((self.black | self.white) & (b << i)) == 0:
                    ret[i] = 1
                    break
        return ret

    cpdef count_arrays(self):
        """

        :return: black_count_array, white_count_array, moves
        """
        cdef int r, c, dr, dc, rr, cc, mine, yours, a, i
        cdef int mark = BLACK
        cdef np.ndarray[SIZE_t, ndim=1] black_count = np.zeros(A + 1, dtype=int)
        cdef np.ndarray[SIZE_t, ndim=1] white_count = np.zeros(A + 1, dtype=int)
        cdef np.ndarray[DOUBLE_t, ndim=1] points = np.zeros(A + 1)
        cdef double point = 1.0
        cdef np.ndarray[SIZE_t, ndim=1] dsr
        cdef np.ndarray[SIZE_t, ndim=1] dsc
        cdef int D = 4
        cdef int moves = popcnt(self.black) + popcnt(self.white)
        cdef ull b = 1

        dsr = np.array([0, 1, 1, 1])
        dsc = np.array([1, 0, 1, -1])

        for _a in range(A):
            a = A - 1 - _a  # reversed
            points[a] = point
            point /= 4.0

        for r in range(BH):
            for c in range(BW):
                for d in range(D):
                    dr = dsr[d]
                    dc = dsc[d]
                    if not (0 <= r + dr * (A - 1) < BH and 0 <= c + dc * (A - 1) < BW):
                        continue
                    mine = 0
                    yours = 0
                    for a in range(A):
                        rr = r + a * dr
                        cc = c + a * dc
                        i = rr * BW + cc
                        if (self.black & (b << i)) > 0:
                            mine += 1
                        if (self.white & (b << i)) > 0:
                            yours += 1
                    if mine > 0 and yours == 0:
                        black_count[mine] += 1
                    if mine == 0 and yours > 0:
                        white_count[yours] += 1

        return black_count, white_count, moves
