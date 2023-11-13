if __name__=="__main__":
    import time, random
    import numpy as np

    from pmatrix import DMatrix, DVec
    from pmatrix.sparse import CSC, CSR, DIA
    random.seed(69)
    shape = (500, 500)
    random_data =  [random.uniform(0.0, 10.0) for i in range(shape[0] * shape[1])]# random.choices([0,1], weights=[8,1], k=100_000_000)
    # random_data2 = random.choices([0,1], weights=[8,1], k=10000)
    t1 = time.time()
    X = DMatrix(random_data).reshape(shape)
    print(time.time()- t1)
    t1 = time.time()
    X_np = np.array(random_data).reshape(shape)
    print(time.time()- t1)
    t1 = time.time()
    A = X @ X.T
    print(time.time()- t1)
    t1 = time.time()
    B = X_np @ X_np.T
    print(time.time()- t1)
    t1 = time.time()
    A_np = np.array(A.tolist())
    print(time.time()- t1)
    t1 = time.time()
    print(A_np - B)
    print(np.max(np.abs(A_np - B)))
    # print(A.tolist())
    # print(B)
    # print("XC", time.time() - t1)
    # t1 = time.time()

    
