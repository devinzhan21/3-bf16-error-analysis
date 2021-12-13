import numpy as np
import tensorflow as tf
float64_a = np.random.random_sample((8, 8))
float32_a = float64_a.astype(np.float32)

bfloat16_a=tf.cast(float32_a,dtype=tf.bfloat16)
#print(bfloat16_a.dtype)
#print(bfloat16_a)
float64_b = np.random.random_sample((8, 8))
float32_b = float64_b.astype(np.float32)

bfloat16_b=tf.cast(float32_b,dtype=tf.bfloat16)
c=np.matmul(float64_a,float64_b)
c1=np.matmul(bfloat16_a,bfloat16_b)
c1=c1.astype(np.float64)
error16 = np.linalg.norm(c - c1, ord=2) / np.linalg.norm(c, ord=2)
print(error16)