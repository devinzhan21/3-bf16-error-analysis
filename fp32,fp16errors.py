import numpy as np
#import tensorflow as tf
def errors(n,m,loop):
    count = 0
    error1 = 0
    error2 = 0

    while (count <= loop):
        float64_a = np.random.random_sample((n, m))
        #print(float64_a.dtype)
        # print(float64_a)
        float32_a = float64_a.astype(np.float32)
        #print(float32_a.dtype)
        float32_a.astype(np.float64)

        float16_a = float64_a.astype(np.float16)
        float16_a.astype(np.float64)
        #print(float16_a.dtype)
        # bias=float64_a-float64_a1
        # print(bias)
        float64_b = np.random.random_sample((n, m))
        # print(float64_b)
        float32_b = float64_b.astype(np.float32)
        float32_b.astype(np.float64)
        c = np.matmul(float64_a, float64_b)
        c1 = np.matmul(float32_a, float32_b)
        # print(c)
        # print(c1)
        float16_b = float64_b.astype(np.float16)
        float16_b.astype(np.float64)

        c2 = np.matmul(float16_a, float16_b)

        error1 += np.linalg.norm(c - c1, ord=2) / np.linalg.norm(c, ord=2)
        error2 += np.linalg.norm(c - c2, ord=2) / np.linalg.norm(c, ord=2)
        count+=1

        errors=np.array([error1/loop,error2/loop])

    return errors



for i in [2,8,32,128,512,1024,2048]:
    print(errors(i,i,100))