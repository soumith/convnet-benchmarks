# coding: utf-8

# # Before start
# 
# There is many important [environment variables](https://mxnet.readthedocs.org/en/latest/env_var.html) which will influence the performance. Change these variable will change the parallelism, memory cost.
# 
# sample command: 
# ```
# MXNET_GPU_WORKER_NTHREADS=4 MXNET_EXEC_NUM_TEMP=4 python3 googlenet.py
# ```
# 
# Speed and memory cost may change due to different level of parallelism

# In[1]:

import mxnet as mx
import numpy as np
import time


# In[2]:

# Basic Info
dev = mx.gpu()
batch_size = 128
dshape = (batch_size, 3, 224, 224)
lshape = (batch_size)
num_epoch = 100

# Mock data iterator
tmp_data = np.random.uniform(-128, 128, dshape).astype("float32")
tmp_label = np.random.uniform(0, 1000, lshape).astype("int").astype("float32")

train_iter = mx.io.NDArrayIter(data=tmp_data, label=tmp_label, batch_size=batch_size, shuffle=False, last_batch_handle='pad')



# GoogLeNet V1: Converted from [Caffe](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt) directly

def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    act = mx.symbol.Activation(data=conv, act_type='relu', name='relu_%s%s' %(name, suffix))
    return act

def InceptionFactory(data, num_1x1, num_3x3red, num_3x3, num_d5x5red, num_d5x5, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd5x5r = ConvFactory(data=data, num_filter=num_d5x5red, kernel=(1, 1), name=('%s_double_3x3' % name), suffix='_reduce')
    cd5x5 = ConvFactory(data=cd5x5r, num_filter=num_d5x5, kernel=(5, 5), pad=(2, 2), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd5x5, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

data = mx.sym.Variable("data")
conv1 = ConvFactory(data, 64, kernel=(7, 7), stride=(2,2), pad=(3, 3))
pool1 = mx.sym.Pooling(conv1, kernel=(3, 3), stride=(2, 2), pool_type="max")
conv2 = ConvFactory(pool1, 64, kernel=(1, 1), stride=(1,1))
conv3 = ConvFactory(conv2, 192, kernel=(3, 3), stride=(1, 1), pad=(1,1))
pool3 = mx.sym.Pooling(conv3, kernel=(3, 3), stride=(2, 2), pool_type="max")

in3a = InceptionFactory(pool3, 64, 96, 128, 16, 32, "max", 32, name="in3a")
in3b = InceptionFactory(in3a, 128, 128, 192, 32, 96, "max", 64, name="in3b")
pool4 = mx.sym.Pooling(in3b, kernel=(3, 3), stride=(2, 2), pool_type="max")
in4a = InceptionFactory(pool4, 192, 96, 208, 16, 48, "max", 64, name="in4a")
in4b = InceptionFactory(in4a, 160, 112, 224, 24, 64, "max", 64, name="in4b")
in4c = InceptionFactory(in4b, 128, 128, 256, 24, 64, "max", 64, name="in4c")
in4d = InceptionFactory(in4c, 112, 144, 288, 32, 64, "max", 64, name="in4d")
in4e = InceptionFactory(in4d, 256, 160, 320, 32, 128, "max", 128, name="in4e")
pool5 = mx.sym.Pooling(in4e, kernel=(3, 3), stride=(2, 2), pool_type="max")
in5a = InceptionFactory(pool5, 256, 160, 320, 32, 128, "max", 128, name="in5a")
in5b = InceptionFactory(in5a, 384, 192, 384, 48, 128, "max", 128, name="in5b")
pool6 = mx.sym.Pooling(in5b, kernel=(7, 7), stride=(1,1), pool_type="avg")
flatten = mx.sym.Flatten(data=pool6)
loss3_classifier = mx.sym.FullyConnected(data=flatten, num_hidden=1000)


# In[4]:

# bind to get executor
# This is what happened behind mx.model.Feedforward
g_exec = loss3_classifier.simple_bind(ctx=dev, grad_req="write", data=dshape)
print("Temp Space: ", g_exec.debug_str().split('\n')[-3])
# Find where to set data


# In[5]:

# data structues 
arg_names = loss3_classifier.list_arguments()
arg_map = dict(zip(arg_names, g_exec.arg_arrays))
grad_map = dict(zip(arg_names, g_exec.grad_arrays))


param_blocks = [(i, arg_map[arg_names[i]], grad_map[arg_names[i]]) for i in range(len(arg_names)) if grad_map[arg_names[i]] != None]
input_ndarray = arg_map["data"]
#label_ndarray = arg_map["prob_label"]
grad = mx.nd.zeros((batch_size, 1000), ctx=mx.gpu())
param_len = len(param_blocks)


# In[6]:

#init
for i in range(param_len):
    param_blocks[i][1][:] = mx.rnd.uniform(-0.01, 0.01, param_blocks[i][1].shape)
    param_blocks[i][2][:] = 0.
# Set data
train_iter.reset()
dbatch = train_iter.next()
dbatch.data[0].copyto(input_ndarray)
#dbatch.label[0].copyto(label_ndarray)
# block all async all
mx.nd.waitall()


# In[ ]:

# Test forward
def test_forward(model, epoch):
    tic = time.time()
    for i in range(epoch):
        model.forward(is_train=True)
        # Note: This command will force thread engine block, which hurts performance a lot
        # Remove it will bring parallelism bias
        model.outputs[0].wait_to_read()
    toc = time.time()
    return (toc - tic) / epoch

print("Avg forward per batch: ", test_forward(g_exec, num_epoch))


# In[ ]:

# Test full path
def test_full(model, epoch):
    tic = time.time()
    for i in range(epoch):
        model.forward(is_train=True)
        model.backward([grad])
        # mock update, prevent NaN
        for i in range(param_len):
            param_blocks[i][1][:] -= 0.0 * param_blocks[i][2]
    # Note: This command will force thread engine block, which hurts performance a lot
    mx.nd.waitall()
    toc = time.time()
    return (toc - tic) / epoch

print("Avg fullpath per batch: ", test_full(g_exec, num_epoch))

