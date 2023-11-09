"""
Diagonal sparse matrix.
"""


from __future__ import annotations
from ._cbase import CBase
from ._svec import SVec
import collections, operator, bisect

class DIA(CBase):
    """
    Diagonal sparse matrix, this matrix can be initialized as:
    DIA(shape=(M,N), cons_diags=[(offset, constant),...], repeat_diags=[(offset, patern),...])
        offset is an integer which counts the offset of the main diagonal. Where positive 
        integers are to the right.
        patern is a list containing a patern that gets wrapped when the end is reached.
    """
    def __init__(self, shape, cons_diags = None, repeat_diags= None, dtype=None) -> DIA:
        CBase.__init__(self)
        self._format = "DIA"
        self.shape = shape
        
        if not isinstance(shape, tuple) or not isinstance(shape[0], int) or not isinstance(shape[1], int):
            raise ValueError("Shape needs to be a length 2 tuple of integers")
        
        data = []

        if cons_diags:
            if any([not isinstance(cd, tuple) for cd in cons_diags]) or any([len(cd)!=2 for cd in cons_diags]) or any([not isinstance(cd[0],int) for cd in cons_diags]): 
                raise ValueError("Constant diagonals are given by tuples with (offset(int), constant)")
            if dtype is None:
                self.dtype = type(cons_diags[0][1])
            try:
                cons_diags = [(o, self.dtype(c))for o, c in cons_diags]
            except Exception as e:
                raise ValueError(f"Inconsistent data type in constant diagonals, {e}")
                
            data += cons_diags
        
        if repeat_diags:
            if any([not isinstance(cd, tuple) for cd in repeat_diags]) or any([len(cd)!=2 for cd in repeat_diags]) or any([not isinstance(cd[0],int) or not isinstance(cd[1],list) for cd in repeat_diags]):
                raise ValueError("Repeat diagonals are given by tuples with (offset(int), repeating_list)")
            if dtype is None:
                self.dtype = type(repeat_diags[0][1][0])
            try:
                repeat_diags = [(o, list(map(self.dtype, patern))) for o, patern in repeat_diags]
            except Exception as e:
                raise ValueError(f"Inconsistent data type in constant diagonals, {e}")
            
            data += repeat_diags

        counted_dict = collections.Counter([d[0] for d in data])
        if any([count > 1 for count in counted_dict.values()]):
            raise ValueError(f"Cannot use 2 values on same diagonal(s) {[k for k,v in counted_dict.items() if v>1]}")
        
        data = sorted(data, key=lambda d: d[0])
        self.offsets = [d[0] for d in data]
        self.data = [d[1] for d in data]
        

    def __add__(self, other) -> DIA:
        return self.__match_operator(other, operator.__add__)
    
    __radd__ = __add__
    
    def __sub__(self, other) -> DIA:
        return self.__match_operator(other, operator.__sub__)
    
    __rsub__ = __sub__
    
    def __match_operator(self, other, opp) -> DIA:
        if not isinstance(other, DIA): return NotImplemented
        if self.shape != other.shape: raise ValueError("Cannot do operation on mismatching shapes.")
        
        for offset, other_d in zip(other.offsets, other.data):
            if offset in self.offsets:
                index = self.offsets.index(offset)
                if isinstance(other_d, list) and isinstance(self.data[index], list): 
                    raise ValueError("Cannot do operation on paterns of different lengths")
                if isinstance(other_d, list):
                    self.data[index] = [opp(self.data[index], d) for d in other_d]
                elif isinstance(self.data[index], list):
                    self.data[index] = [opp(d,other_d) for d in self.data[index]]
                else:
                    self.data[index] = opp(self.data[index], other_d)
            
            else:
                index = bisect.bisect_left(self.offsets, offset)
                self.offsets.insert(index, offset)
                if isinstance(other_d, list):
                    self.data.insert(index, [opp(0, d) for d in other_d])
                else:
                    self.data.insert(index, opp(0, other_d))
        
        return self
        
    def get_row_data(self, row_i):
        row_d = []
        for index, data in zip([offset + row_i for offset in self.offsets], self.data):
            if index < 0 or index >= self.shape[1]: continue
            if isinstance(data, list):
                data = data[min(row_i, index) % len(data)]
            row_d.append((index, data))
        return row_d
    
    def get_col_data(self, col_i):
        col_d = []
        for index, data in zip([col_i - offset  for offset in self.offsets], self.data):
            if index < 0 or index >= self.shape[0]: continue
            if isinstance(data, list):
                data = data[min(col_i, index) % len(data)]
            col_d.append((index, data))
        col_d.reverse()
        return col_d
    
    def _to_svecs(self):
        return [SVec(self.get_row_data(i), dtype=self.dtype, orientation='r', length=self.shape[1]) for i in range(self.shape[0])]

    def _to_full_data(self):
        return [svec.to_dvec() for svec in self._to_svecs()]