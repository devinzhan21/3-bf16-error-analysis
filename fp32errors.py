import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
def errors(n,m,loop):
    count = 0
    error1 = 0


    while (count <= loop):
        #float64_a = np.random.random_sample((n, m))
        float64_a =np.random.exponential(scale=10.0, size=(n,m)) 
        #指数分布
        #print(float64_a.dtype)
        # print(float64_a)
        float32_a = float64_a.astype(np.float32)
        #print(float32_a.dtype)
        float32_a.astype(np.float64)


        #print(float16_a.dtype)
        # bias=float64_a-float64_a1
        # print(bias)
        #float64_b = np.random.random_sample((n, m))
        float64_b =np.random.exponential(scale=10.0, size=(n,m))
        #指数分布
        # print(float64_b)
        float32_b = float64_b.astype(np.float32)
        float32_b.astype(np.float64)
        c = np.matmul(float64_a, float64_b)
        c1 = np.matmul(float32_a, float32_b)
        # print(c)
        # print(c1)




        error1 += np.linalg.norm(c - c1, ord=2) / np.linalg.norm(c, ord=2)

        count+=1

        errors=error1/loop

    return errors


y_data=np.zeros(shape=(1,30))


for i in range(30):
    y_data[0,i]=errors(4*(i+1),4*(i+1),200)
   

print(y_data)

