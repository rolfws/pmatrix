import operator

class LogicCore:
    """
    This handles some global logic
    """
    def _parse_getkey(self, key):
        def parse_part(key_part, axis):
            if isinstance(key_part, int):
                if key_part >= self.shape[axis]: raise IndexError(f"Index {key_part} out of range for axis {axis} with length {self.shape[axis]}")
                return key_part
            if isinstance(key_part, slice):
                return key_part
            elif isinstance(key_part, (list, tuple)):
                if any([not isinstance(k, int) for k in key_part]): raise ValueError(f"Can only index with list of int")
                if any([k>= self.shape[axis] for k in key_part]): raise ValueError(f"IndexError {key_part} out of range for axis {axis} with length {self.shape[axis]}")
                return operator.itemgetter(*key_part)
            else:
                raise ValueError(f"Cannot index using {type(key_part)}, only int, slice or list of int are allowed.")
            
        def fix_size_one(key):
            i = 0
            fixed_key = []
            for s in self.shape:
                if s == 1:
                    fixed_key.append(slice(None))
                else:
                    fixed_key.append(key[i])
                    i += 1
            return fixed_key

        if isinstance(key, tuple):
            if len(key) > sum(s > 1 for s in self.shape): raise ValueError(f"Matrix is {sum(s > 1 for s in self.shape)}D, while {len(key)} are indexed")
            key = fix_size_one(key)
            parsed_key = tuple(parse_part(key[i], i) if i < len(key) else slice(None) for i in range(len(self.shape)))
            
        else:
            index = next(x for x, val in enumerate(self.shape) if val > 1)
            parsed_key = tuple([None] * index + [parse_part(key, index)] + [slice(None)] * (len(self.shape) - index - 1))
        return parsed_key
    
    def _get_item_logic(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return self.data[key]
        elif isinstance(key, list):
            return list(operator.itemgetter(*key)(self.data))
        elif isinstance(key, operator.itemgetter):
            return list(key(self.data))
        elif isinstance(key, tuple):
            raise ValueError(f"Vector is 1D, can not use tuple")
        else:
            raise ValueError(f"Can only slice vector using, int, slice or list of int")
        
import time
class Timer:
    def __init__(self) -> None:
        self.t = time.time()
        self.timings = {}
        self.rounds = {}
    
    def time(self, name):
        self.timings[name] = self.timings.get(name, 0) + time.time() - self.t
        self.rounds[name] = self.rounds.get(name, 0) + 1
        self.t = time.time()

    def add_timer(self, other):
        for (k, timing), round in zip(other.timings.items(), other.rounds.values()):
            self.timings[k] = self.timings.get(k, 0) + timing
            self.rounds[k] = self.rounds.get(k, 0) + round

    def __str__(self):
        str_dict = {
            k: {
                "total":v,
                "rounds": self.rounds[k],
                "average": v/self.rounds[k]
            }
            for k, v in self.timings.items()
            if k != "reset"
        }
        return str_dict.__str__()