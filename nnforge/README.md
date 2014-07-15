For compiling nnForge, look at INSTALL.md

nnForge Convolution kernel (kepler) is at: https://github.com/milakov/nnForge/blob/master/nnforge/cuda/convolution_layer_updater_cuda_kepler.cuh

Quote from Maxim:
> I am sorry to say that it is not an easy task at all. The actual convolution is done in https://github.com/milakov/nnForge/blob/master/nnforge/cuda/convolution_layer_tester_cuda_kepler.cuh but you cannot just call enqueue_test and sync on the stream, you would need to do a lot of preparation calls.

So, I guess I'll tackle this last, seems like a complicated task.
