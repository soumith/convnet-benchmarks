require 'nn'
require 'sys'
require 'cunn'
require 'ccn2'
 
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')
 
 
steps = 4 -- nb of steps in loop to average perf
ops = 2 -- ops per point

runs = {
   {
      ni = 3,
      no = 96,
      kw = 11,
      kh = 11,
      iw = 128,
      ih = 128,
      bs = 128,
      dw = 1,
      dh = 1,
   },
   {
      ni = 64,
      no = 128,
      kw = 9,
      kh = 9,
      iw = 64,
      ih = 64,
      bs = 128,
      dw = 1,
      dh = 1,
   },
   {
      ni = 128,
      no = 128,
      kw = 9,
      kh = 9,
      iw = 32,
      ih = 32,
      bs = 128,
      dw = 1,
      dh = 1,
   },
   {
      ni = 128,
      no = 128,
      kw = 7,
      kh = 7,
      iw = 16,
      ih = 16,
      bs = 128,
      dw = 1,
      dh = 1,
   },
}

for i,run in ipairs(runs) do
   -- params for run:
   local ni,no,kw,kh,bs,iw,ih,dw,dh = run.ni,run.no,run.kw,run.kh,run.bs,run.iw,run.ih,run.dw,run.dh
   print('')
   print('CONFIG: input = ' .. ni..'x'..iw..'x'..ih..' * ker = ' .. ni..'x'..no..'x'..kw..'x'..kh .. ' (bs = '..bs..', stride = ' .. dw .. ')')

   n1 = nn.SpatialConvolutionCUDA(ni,no,kw,kh,dw,dh):cuda()
   n2 = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh):cuda()
   n3 = ccn2.SpatialConvolution(ni,no,kw,dw):cuda()

   i1 = torch.randn(ni, ih, iw, bs):cuda()
   i2 = torch.randn(bs, ni, ih, iw):cuda()
   i3 = torch.randn(ni, ih, iw, bs):cuda()

   o1 = n1:forward(i1)
   o2 = n2:forward(i2)
   o3 = n2:forward(i3)

   cutorch.synchronize()
   sys.tic()
   for t = 1,steps do
      o1 = n1:updateOutput(i1)
   end
   cutorch.synchronize()
   tm = sys.toc()/steps
   print('nn.SpatialConvolutionCUDA: ' .. (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) /dw/dh * bs * ops / tm / 1e9) .. ' GFLOP/s (tm = ' .. tm .. ')')

   cutorch.synchronize()
   sys.tic()
   for t = 1,steps do
      o2 = n2:updateOutput(i2)
   end
   cutorch.synchronize()
   tm = sys.toc()/steps
   print('nn.SpatialConvolutionMM: ' .. (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) /dw/dh * bs * ops / tm / 1e9) .. ' GFLOP/s (tm = ' .. tm .. ')')

   cutorch.synchronize()
   sys.tic()
   for t = 1,steps do
      o3 = n3:updateOutput(i3)
   end
   cutorch.synchronize()
   tm = sys.toc()/steps
   print('ccn2.SpatialConvolution: ' .. (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) /dw/dh * bs * ops / tm / 1e9) .. ' GFLOP/s (tm = ' .. tm .. ')')

   collectgarbage()
end

