from .._core._dvec import DVec
import operator

class SVec:
    def __init__(self, data, dtype=None, orientation='r', length=None):
        self._format = "svec"

        if isinstance(data, DVec):
            self.from_dvec(data)
            return

        self.orientation = orientation
        if not isinstance(data, list):
            raise TypeError(f"Vector data must be a list")
        
        if len(data)==0 and length is None:
            raise ValueError("Can not initiate an empty sparse vector without a length")
        
        elif len(data)==0:
            self.data = []
            self.length = length
            if dtype is None:
                dtype = int
            self.dtype = dtype

        elif isinstance(data[0], tuple):
            self.from_indexed(data, length=length)

        else:
            self.from_dense(data, dtype=dtype)


    def from_dvec(self, vec:DVec):
        self.dtype = vec.dtype
        self.length = vec.length
        self.orientation = vec.orientation
        self.data = [(i, d) for i, d in enumerate(vec) if d]

    def from_dense(self, data, dtype=None):
        if dtype is None:
            dtype = type(data[0])

        self.dtype = dtype

        if not dtype in (int, float, complex, bool): raise TypeError(f"Data must be of type int, float or complex, not {dtype}")

        try:
            data = list(map(self.dtype, data))
        except (ValueError, TypeError)  as e:
            raise TypeError(f"Not all data can be converted into {self.dtype.__name__}, {e}")
        
        self.data = [(i, d) for i, d in enumerate(data) if d]
        self.length = len(data)
    
    def from_indexed(self, data, length):
        if length is None:
            raise ValueError(f"Unkown length")
        self.dtype = type(data[0][1])
        if not self.dtype in (int, float, complex, bool): raise TypeError(f"Data must be of type int, float or complex, not {self.dtype}")
        try:
            data = [(i, self.dtype(d)) for i, d in data]
        except (ValueError, TypeError)  as e:
            raise TypeError(f"Not all data can be converted into {self.dtype.__name__}, {e}")
    
        self.data = data
        self.length = length

    def get_row_data(self, row_i):
        if self.orientation == 'r' and row_i != 0:
            raise IndexError("Row vector only has 1 row")
        elif self.orientation == 'r':
            return self.data
        else:
            return [d for d in self.data if d[0] == row_i]
        
    def get_col_data(self, col_i):
        if self.orientation == 'c' and col_i != 0:
            raise IndexError("column vector only has 1 column")
        elif self.orientation == 'c':
            return self.data
        else:
            return [d for d in self.data if d[0] == col_i]
    
    def __mul__(self, other):
        return self.__match_operator(other, operator.__mul__)
    
    def __matmul__(self, other):
        return sum([d[1] for d in (self * other).data])

    def __match_operator(self, other, opp):
        if not isinstance(other, SVec):
            return NotImplemented
        if self.length != other.length:
            raise IndexError(f"Length {self.length} and {other.length} are not equal:")
        i, j, opp_list = 0, 0, []
        while i < len(self.data) and j < len(other.data):
            if self.data[i][0] == other.data[j][0]:
                opp_list.append((self.data[i][0], opp(self.data[i][1], other.data[j][1])))
                i += 1
                j += 1
            elif self.data[i][0] > other.data[j][0]:
                j += 1
            else:
                i +=1
        return SVec(opp_list, orientation=self.orientation, length=self.length)
    
    def sum(self):
        return sum([d[1] for d in self.data])

    def to_dvec(self):
        dense_data = [0] * self.length
        for i, d in self.data:
            dense_data[i] = d
        return DVec(dense_data, dtype=self.dtype, orientation=self.orientation)
    
    def __str__(self):
        return self.data.__str__()
