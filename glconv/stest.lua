require 'nn'
require 'sys'
require 'cutorch'
print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

gl = require 'libglconv'
gl.logging(1)
gl.useintp(1)
--gl.precision(1)	-- required on NVidia
precision = 1
torch.setdefaulttensortype('torch.FloatTensor')

steps = 10 -- nb of steps in loop to average perf

runs = {
   
   {
      -- first layer
      ni = 128,
      no = 128,
      kw = 3,
      kh = 3,
      iw = 128,
      ih = 128,
      bs = 128,
      dh = 1,
      dw = 1
   },
}

function nn.SpatialConvolutionMM:updateOutput(input)
	gl.precision(precision)
	if self.weight:dim() == 2 then
		self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
	end
	gl.conv(input, self.weight, self.output, self.bias)
	return self.output
end

for i,run in ipairs(runs) do
   -- params for run:
   local ni,no,kw,kh,bs,iw,ih,dw,dh = run.ni,run.no,run.kw,run.kh,run.bs,run.iw,run.ih,run.dw,run.dh
   print('')
   print('CONFIG: input = ' .. ni..'x'..iw..'x'..ih..' * ker = ' .. ni..'x'..no..'x'..kw..'x'..kh 
	    .. ' (bs = '..bs..', stride = ' .. dw .. ')')
   collectgarbage()
   local input = torch.randn(bs,ni,ih,iw)
   local network = nn.SpatialConvolutionMM(ni, no, kw, kh)
   local output = network:forward(input)

   sys.tic()
   for t = 1,steps do
      output = network:forward(input)
   end
   local tmf = sys.toc()/steps
   print(string.format("%-30s %25s %10.2f", 'glconv', ':updateOutput():', tmf*1000))
end
