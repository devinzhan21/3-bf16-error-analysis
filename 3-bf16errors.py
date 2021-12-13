import numpy as np
import tensorflow as tf


class fp32to3bf16:

    bfloat16_0=0
    bfloat16_1=0
    bfloat16_2=0

    def __init__(self, float32_a):
        self.float32_a=float32_a
        self.bfloat16_0 = tf.cast(float32_a, dtype=tf.bfloat16)
        float32_a0 = tf.cast(self.bfloat16_0, dtype=tf.float32)
        float32_a1 = float32_a - float32_a0
        self.bfloat16_1 = tf.cast(float32_a1, dtype=tf.bfloat16)
        float32_a2 = float32_a1 - tf.cast(self.bfloat16_1, dtype=tf.float32)
        self.bfloat16_2 = tf.cast(float32_a2, dtype=tf.bfloat16)


def bf16matmul(a, b):

    bf_a=fp32to3bf16(a)
    bf_b=fp32to3bf16(b)
    result=tf.cast(np.matmul(bf_a.bfloat16_0,bf_b.bfloat16_0),dtype=tf.float64)+\
           tf.cast(np.matmul(bf_a.bfloat16_0,bf_b.bfloat16_1),dtype=tf.float64)+\
           tf.cast(np.matmul(bf_a.bfloat16_1,bf_b.bfloat16_0),dtype=tf.float64)+\
           tf.cast(np.matmul(bf_a.bfloat16_1,bf_b.bfloat16_1),dtype=tf.float64)+\
           tf.cast(np.matmul(bf_a.bfloat16_2,bf_b.bfloat16_0),dtype=tf.float64)+\
           tf.cast(np.matmul(bf_a.bfloat16_0,bf_b.bfloat16_2),dtype=tf.float64)



    return result
def bf16_3error(n,m,loop):
    count=0
    errors3_16=0
    while(count<=loop):
        float64_a = np.random.random_sample((n, m))
        float32_a = float64_a.astype(np.float32)
        float64_b = np.random.random_sample((n, m))
        float32_b = float64_b.astype(np.float32)
        bfloat_ab = bf16matmul(float32_a, float32_b)
        c = np.matmul(float64_a, float64_b)
        c1 = tf.cast(bfloat_ab, dtype=tf.float64)
        errors3_16+= np.linalg.norm(c - c1, ord=2) / np.linalg.norm(c, ord=2)
        count+=1
        error3_16=errors3_16/loop
    return error3_16


for i in [2,8,32,128]:
    print(bf16_3error(i,i,100))












