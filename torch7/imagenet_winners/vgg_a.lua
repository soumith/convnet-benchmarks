function vgg_a(lib)
   local SpatialConvolution = lib[1]
   local SpatialMaxPooling = lib[2]
   local ReLU = lib[3]
   local SpatialZeroPadding = nn.SpatialZeroPadding
   local padding = true
   local stride1only = false
   if lib[5] == 'fbfft' then
      padding = false -- fbfft does not support implicit zero padding
      stride1only = true -- fbfft does not support convolutions that are not stride-1
   end

   local modelType = 'A' -- on a titan black, B/D/E run out of memory even for batch-size 32

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   if modelType == 'A' then
      cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'B' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'D' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'}
   elseif modelType == 'E' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end

   local features = nn.Sequential()
   do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            features:add(SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1))
            features:add(ReLU(true))
            iChannels = oChannels;
         end
      end
   end

   features:get(1).gradInput = nil

   local classifier = nn.Sequential()
   classifier:add(nn.View(512*7*7))
   classifier:add(nn.Linear(512*7*7, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   -- classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   -- classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 1000))
   -- classifier:add(nn.LogSoftMax())

   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model,'VGG Model-' .. modelType, {64,3,224,224}
end

return vgg_a
