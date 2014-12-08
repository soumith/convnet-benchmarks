function alexnet(lib)
   local SpatialConvolution = lib[1]
   local SpatialMaxPooling = lib[2]
   local ReLU = lib[3]

   local features = nn.Concat(2)
   local branch1 = nn.Sequential() -- branch 1
   branch1:add(SpatialConvolution(3,48,11,11,4,4,2,2))       -- 224 -> 55
   branch1:add(ReLU())
   branch1:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   branch1:add(SpatialConvolution(48,128,5,5,1,1,2,2))       --  27 -> 27
   branch1:add(ReLU())
   branch1:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   branch1:add(SpatialConvolution(128,192,3,3,1,1,1,1))      --  13 ->  13
   branch1:add(ReLU())
   branch1:add(SpatialConvolution(192,192,3,3,1,1,1,1))      --  13 ->  13
   branch1:add(ReLU())
   branch1:add(SpatialConvolution(192,128,3,3,1,1,1,1))      --  13 ->  13
   branch1:add(ReLU())
   branch1:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

   local branch2 = branch1:clone() -- branch 2, if using in real-life, reset weights of this branch
   features:add(branch1)
   features:add(branch2)
   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   -- classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   -- classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, 1000))
   -- classifier:add(nn.LogSoftMax())

   local model = nn.Sequential()
   model:add(features):add(classifier)
   
   return model,'AlexNet',{128,3,224,224}
end

return alexnet
