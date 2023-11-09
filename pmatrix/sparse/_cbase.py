from .._core._dmatrix import DMatrix
from .._core._dvec import DVec
from .._core._logiccore import LogicCore
from ._svec import SVec

class CBase(LogicCore):
    def __init__(self):
        self._shape = None
        self._format = "und"

        if self.__class__.__name__ == 'Base':
            raise ValueError("This class is not intended"
                             " to be instantiated directly.")

    def __matmul__(self, other)-> DMatrix:
        if not hasattr(other, "_format"): return NotImplemented
        if self.shape[1] != other.shape[0]: raise ValueError(f"Can not do a dot product with between matrices with size {self.shape} and {other.shape}")
        if not hasattr(self, "get_row_data") or not hasattr(other, "get_col_data"): raise NotImplementedError
        row_d = [SVec(self.get_row_data(i), dtype=self.dtype, orientation='r', length=self.shape[1]) for i in range(self.shape[0])]
        col_d = [SVec(other.get_col_data(i), dtype=other.dtype, orientation='c', length=other.shape[0]) for i in range(other.shape[1])]
        return DMatrix(data = [[row @ col for col in col_d] for row in row_d])
    
    def __rmatmul__(self, other) -> DMatrix:
        if not hasattr(other, "_format"): return NotImplemented
        if other.shape[1] != self.shape[0]: raise ValueError(f"Can not do a dot product with between matrices with size {other.shape} and {self.shape}")
        if not hasattr(other, "get_row_data") or not hasattr(self, "get_col_data"): raise NotImplementedError
        row_d = [SVec(other.get_row_data(i), dtype=other.dtype, orientation='r', length=other.shape[1]) for i in range(other.shape[0])]
        col_d = [SVec(self.get_col_data(i), dtype=self.dtype, orientation='c', length=self.shape[0]) for i in range(self.shape[1])]
        return DMatrix(data = [[row @ col for col in col_d] for row in row_d])
    
    def to_dmatrix(self) -> DMatrix:
        if not hasattr(self, "_to_full_data"): raise NotImplementedError
        data = self._to_full_data()
        return DMatrix(data=data)
    
    def __match_operator(self, other):
        if 1 in other.shape:
            pass
        elif 1 in self.shape:
            pass
        else:
            if self.shape != other.shape: raise ValueError("Cannot do operation on mismatching shapes.")

    def __getitem__(self, key):
        pass

    
    @classmethod
    def from_dmatrix(self, matrix):
        if not isinstance(matrix, DMatrix): raise ValueError(f"Matrix is not of type DMatrix")
        if not hasattr(self, "self_from_svecs"): raise NotImplementedError

        matrix._force_orientation(self._orientation)

        if isinstance(matrix.data, DVec):
            svecs = [SVec(matrix.data)]
        else:
            svecs = [SVec(vec) for vec in matrix.data]
        
        if not hasattr(self, "self_from_svecs"): raise NotImplementedError
        return self.self_from_svecs(svecs, matrix.shape)


    def __str__(self):
        return self.to_dmatrix().__str__()