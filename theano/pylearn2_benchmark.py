import numpy as np
import theano
import pylearn2
from pylearn2.models.mlp import ConvElemwise, ConvNonlinearity

 
steps = 4 # nb of steps in loop to average perf
ops = 2 # ops per point

runs = [
   {
      'ni' : 3,
      'no' : 96,
      'kw' : 11,
      'kh' : 11,
      'iw' : 128,
      'ih' : 128,
      'bs' : 128,
      'dw' : 1,
      'dh' : 1,
   },
   {
      'ni' : 64,
      'no' : 128,
      'kw' : 9,
      'kh' : 9,
      'iw' : 64,
      'ih' : 64,
      'bs' : 128,
      'dw' : 1,
      'dh' : 1,
   },
   {
      'ni' : 128,
      'no' : 128,
      'kw' : 9,
      'kh' : 9,
      'iw' : 32,
      'ih' : 32,
      'bs' : 128,
      'dw' : 1,
      'dh' : 1,
   },
   {
      'ni' : 128,
      'no' : 128,
      'kw' : 7,
      'kh' : 7,
      'iw' : 16,
      'ih' : 16,
      'bs' : 128,
      'dw' : 1,
      'dh' : 1,
   }
]

for i in range(4):
   run = runs[i]
   # params for run:
   ni,no,kw,kh,bs,iw,ih,dw,dh = run['ni'],run['no'],run['kw'],run['kh'],run['bs'],run['iw'],run['ih'],run['dw'],run['dh']
   print ''
   print 'CONFIG: input =',ni,'x',iw,'x',ih,'* ker =',ni,'x',no,'x',kw,'x',kh,'(bs =',bs,', stride =',dw,')'

   conv = ConvElemwise(no,(kw,kh),'ConvTest', ConvNonlinearity())

   inputBatch = np.rand.randn(bs, ni, ih, iw)
   sharedX = theano.sandbox.cuda.shared(x)
   sharedX:set_value(inputBatch)
   
   params = conv:get_params()
   
   X = theano.tensor.tensor4()
   
   Y=conv.fprop(X)
   
   fprop = theano.function(X,Y,givens=[(X,sharedX)])

   sys.tic()
   for i in range(steps):
      o1 = fprop(X)
   
   
   tm = sys.toc()/steps
   print('pylearn2.models.mlp.ConvElemwise: ' .. (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) /dw/dh * bs * ops / tm / 1e9) .. ' GFLOP/s (tm = ' .. tm .. ')')
