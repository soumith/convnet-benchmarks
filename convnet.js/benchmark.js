var convnetjs = require("./convnetjs/build/convnet.js");

// L1 Conv Layer definition
var opt = { in_sx:128, in_sy:128, in_depth:3, sx:11, filters:96, stride: 1, pad: 0 };
var layer = new convnetjs.ConvLayer(opt);

// create a random input volume
var x = new convnetjs.Vol(128, 128, 3);

// run it through batch_size number of times
var batch_size = 128;
var dtall = 0;
for(var i=0;i<batch_size;i++) { // batch of 128
  var t0 = +new Date();
  layer.forward(x); // forward
  var t1 = +new Date();
  var dt = t1 - t0;
  dtall += dt;
  console.log(i + ' took ' + dt + 'ms. Estimating full batch to take ' + (dtall/(i+1))*batch_size + 'ms');
}
console.log('total: ' + dtall + 'ms');
