#include <iostream>
#include <stdio.h>

#include <nnforge/cuda/cuda.h>

#include <nnforge/nnforge.h>

int main(int argc, char* argv[])
{
	try
	{
		nnforge::cuda::cuda::init();		
		nnforge::convolution_layer layer(std::vector<unsigned int>(2, 11), 3, 96);

		nnforge::layer_configuration_specific input_configuration;
		input_configuration.feature_map_count = 3;
		input_configuration.dimension_sizes.push_back(128);
		input_configuration.dimension_sizes.push_back(128);

		float fflops = layer.get_forward_flops(input_configuration);
		float bflops = layer.get_backward_flops(input_configuration);
		float bbflops = layer.get_backward_flops_2nd(input_configuration);
		std::cout << "convolution_layer 3->96 11x11"<< std::endl;
		std::cout << ":forward gflop/s: " << fflops/1000000000 << std::endl;
		std::cout << ":backward gflop/s: " << bflops/1000000000 << std::endl;
		std::cout << ":hessian  gflop/s: " << bbflops/1000000000 << std::endl;

	}
	catch (const std::exception& e)
	{
		std::cout << "Exception caught: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
