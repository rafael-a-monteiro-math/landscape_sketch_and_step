################################################################################
# HOW TO USE
################################################################################
# 

#1) Using subprocess with shell=True, when should
import sys
import numpy as np

INPUT_NAME=sys.argv[1]
OUTPUT_NAME=sys.argv[2]

###### Multi-bump, increasingly difficult heights
y_bump=lambda x, mu, phase:\
    np.sin(mu * x + phase)
y_trend=lambda x, eps, center:\
    eps *  np.linalg.norm(x - center, axis=1)**2
y_func=lambda x:\
    np.ravel(
        (y_bump(x,10*np.pi, +np.pi/2) + 1)/2
        )*(25+y_trend(x,30,.1))+\
            np.ravel(
                (1 - y_bump(x, 10*np.pi, + np.pi/2))/2 
                )*(5 + y_trend (x, 25, .9))
with open (INPUT_NAME+'.npy', 'rb') as f:
    x=np.load(f)
# multi- d
M_aux=np.copy(x)

for i in range(x.shape[1]):
    x_now=np.reshape(x[:,i], (-1,1))
    y=y_func(x_now).astype(np.float32)
    # Normalize
    a=y_func(.1*np.ones((1,1), dtype=np.float32)).astype(np.float32)
    y=y / a
    M_aux[:,i]=y

y=np.product(M_aux, axis=1, keepdims=True)
with open(OUTPUT_NAME+'.npy', 'wb') as f:
    np.save(f, y)
