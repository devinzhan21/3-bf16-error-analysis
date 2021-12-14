import numpy as np
import tensorflow as tf
def bf16error(n,m,loop):
    count=0
    errors16=0
    while(count<=loop):
        float64_a = np.random.random_sample((n, m))
        float32_a = float64_a.astype(np.float32)
        bfloat16_a = tf.cast(float32_a, dtype=tf.bfloat16)
        float32_a1=tf.cast(bfloat16_a,dtype=tf.float32)
        # print(bfloat16_a.dtype)
        # print(bfloat16_a)
        float64_b = np.random.random_sample((n, m))
        float32_b = float64_b.astype(np.float32)
        bfloat16_b = tf.cast(float32_b, dtype=tf.bfloat16)
        float32_b1=tf.cast(bfloat16_b,dtype=tf.float32)

        c = np.matmul(float64_a, float64_b)
        c1 = np.matmul(float32_a1, float32_b1)
        c1 = c1.astype(np.float64)
        errors16 += np.linalg.norm(c - c1, ord=2) / np.linalg.norm(c, ord=2)
        count += 1
        error16 = errors16 / loop

    return (error16)


y_data = np.zeros(shape=(1, 30))

for i in range(30):
    y_data[0, i] = bf16error(4 * (i + 1), 4 * (i + 1), 200)

print(y_data)