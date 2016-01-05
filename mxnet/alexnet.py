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
tmp_data = np.random.uniform(-1, 1, dshape).astype("float32")

train_iter = mx.io.NDArrayIter(data=tmp_data,  batch_size=batch_size, shuffle=False, last_batch_handle='pad')



# In[5]:

def get_alexnet_symbol():
    ## define alexnet
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=64)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
#    lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 2
    conv2 = mx.symbol.Convolution(
        data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=192)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
#    lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 3
    conv3 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    conv5 = mx.symbol.Convolution(
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
    relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
    # stage 5
    fc2 = mx.symbol.FullyConnected(data=relu6, num_hidden=4096)
    relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
    # stage 6
    fc3 = mx.symbol.FullyConnected(data=relu7, num_hidden=1000)
    return fc3

# In[6]:

# bind to get executor
# This is what happened behind mx.model.Feedforward
fc3 = get_alexnet_symbol()
alex_exec = fc3.simple_bind(ctx=dev, grad_req="write", data=dshape)
print("Temp space: ", alex_exec.debug_str().split('\n')[-3])
# Find where to set data


# In[7]:

# some useful structure
# data structues 
arg_names = fc3.list_arguments()
arg_map = dict(zip(arg_names, alex_exec.arg_arrays))
grad_map = dict(zip(arg_names, alex_exec.grad_arrays))


param_blocks = [(i, arg_map[arg_names[i]], grad_map[arg_names[i]]) for i in range(len(arg_names)) if grad_map[arg_names[i]] != None]
input_ndarray = arg_map["data"]
grad = mx.nd.zeros((batch_size, 1000), ctx=mx.gpu())
param_len = len(param_blocks)


# In[8]:

#init
for i in range(param_len):
    param_blocks[i][1][:] = mx.rnd.uniform(-0.01, 0.01, param_blocks[i][1].shape)
    param_blocks[i][2][:] = 0.
# Set data
train_iter.reset()
dbatch = train_iter.next()
dbatch.data[0].copyto(input_ndarray)
# block all async all
mx.nd.waitall()


# In[12]:

# Test forward
def test_forward(model, epoch):
    tic = time.time()
    for i in range(epoch):
        model.forward(is_train=True)
        # Note: This command will force thread engine block, which hurts performance a lot
        # Remove it will bring parallelism bias
        # model.outputs[0].wait_to_read()
    model.outputs[0].wait_to_read()
    toc = time.time()
    return (toc - tic) / epoch

print("Avg forward per batch: ", test_forward(alex_exec, num_epoch))


# In[13]:

# Test full path
def test_full(model, epoch):
    tic = time.time()
    for i in range(epoch):
        model.forward(is_train=True)
        model.backward([grad])
        #model.outputs[0].wait_to_read()
        # mx.nd.waitall()
        # mock update
        for i in range(param_len):
            param_blocks[i][1][:] -= 0.0 * param_blocks[i][2][:]
    # Note: This command will force thread engine block, which hurts performance a lot
    mx.nd.waitall()
    toc = time.time()
    return (toc - tic) / epoch

print("Avg fullpath per batch: ", test_full(alex_exec, num_epoch))


# In[ ]:

