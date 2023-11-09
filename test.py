if __name__=="__main__":
    import time, random

    from pmatrix import DMatrix, DVec
    from pmatrix.sparse import CSC, CSR, DIA
    random_data = random.choices([0,1], weights=[8,1], k=100_000_000)
    random_data2 = random.choices([0,1], weights=[8,1], k=10000)
    t1 = time.time()
    X = DMatrix(random_data).reshape((10000,10000))
    print("X", time.time() - t1)
    t1 = time.time()
    Y = DMatrix(random_data2)
    print("Y", time.time() - t1)
    X @ Y
    print("X@Y", time.time() - t1)
    t1 = time.time()
    # print(X@Y)

    t1 = time.time()
    XR = CSR.from_dmatrix(X)
    print("XR", time.time() - t1)
    t1 = time.time()
    XC = CSC.from_dmatrix(X)
    print("XC", time.time() - t1)
    t1 = time.time()
    XR @ Y
    print("XR@Y", time.time() - t1)
    t1 = time.time()
    Y.T @ XC
    print("Y_t@XC", time.time() - t1)
    t1 = time.time()

    # print("XC", time.time() - t1)
    # t1 = time.time()

    
