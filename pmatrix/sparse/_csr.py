from __future__ import annotations
from ._cbase import CBase
from ._svec import SVec
import time, itertools, bisect

class CSR(CBase):
    _orientation = 'r'
    def __init__(self):
        CBase.__init__(self)
        self._format = "CSR"

    def get_row_data(self, row_i):
        start, stop = self.indptr[row_i], self.indptr[row_i + 1]
        return list(zip(self.indices[start:stop], self.data[start:stop]))

    def get_col_data(self, col_i):
        col_d = [(i,d) for i, (ind, d) in enumerate(zip(self.indices, self.data)) if ind==col_i]
        return [(bisect.bisect_right(self.indptr, i) - 1, d) for i, d in col_d]
    
    def T(self, inplace=False):
        t_data, t_indptr, t_indices = [], [0], []
        t_shape = (self.shape[1], self.shape[0])
        t_dtype= self.dtype
        for i in range(self.shape[0]):
            col_d = self.get_col_data(i)
            t_indptr.append(t_indptr[i] + len(col_d))
            t_data += [d[1] for d in col_d]
            t_indices += [d[0] for d in col_d]
        
        if inplace:
            self_t = self
        else:
            self_t = CSR()

        self_t.indptr = t_indptr
        self_t.indices = t_indices
        self_t.data = t_data
        self_t.shape = t_shape
        self_t.dtype = t_dtype
        return self_t
    
    def _to_svecs(self):
        return [SVec(list(zip(self.indices[start:stop], self.data[start:stop])), dtype=self.dtype, orientation='r', length=self.shape[1]) for start, stop in itertools.pairwise(self.indptr)]

    def _to_full_data(self):
        return [svec.to_dvec() for svec in self._to_svecs()]

    @classmethod
    def self_from_svecs(cls, svecs, shape):
        self = CSR()
        self.shape = shape
        self.dtype = svecs[0].dtype
        self.indices, self.data, self.indptr = [], [], [0]
        for svec in svecs:
            self.indices += [d[0] for d in svec.data]
            self.data += [d[1] for d in svec.data]
            self.indptr.append(len(self.data))
        return self