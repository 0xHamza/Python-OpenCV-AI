#%%

import tensorflow as tf

print("Hello world!")


print(tf.__version__)


x= [[3.]]
y= [[4.]]
print("Result: {}".format(tf.matmul(x,y)))
