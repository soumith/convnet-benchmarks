#include "ccv.h"
#include <ctype.h>

void cwc_bench_runtime(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_convnet_train_param_t params);

int main(int argc, char** argv)
{
	ccv_enable_default_cache();
	assert(argc == 2);
	ccv_categorized_t categorized;
	ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_categorized_t), 64, 0);
	{
		ccv_file_info_t input;
		input.filename = (char*)ccmalloc(1024);
		strncpy(input.filename, argv[1], 1024);
		categorized = ccv_categorized(0, 0, &input);
		ccv_array_push(categorizeds, &categorized);
	}
	/* MattNet parameters */
	ccv_convnet_layer_param_t params[1] = {
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 0,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 128,
					.cols = 128,
					.channels = 3,
					.partition = 1,
				},
			},
			.output = {
				.convolutional = {
					.count = 96,
					.strides = 1,
					.border = 0,
					.rows = 11,
					.cols = 11,
					.channels = 3,
					.partition = 2,
				},
			},
		}
	};
	ccv_convnet_t* convnet = ccv_convnet_new(1, ccv_size(128, 128), params, sizeof(params) / sizeof(ccv_convnet_layer_param_t));
	ccv_convnet_verify(convnet, 1000);
	ccv_convnet_layer_train_param_t layer_params[13];
	memset(layer_params, 0, sizeof(layer_params));
	int i;
	for (i = 0; i < 13; i++)
	{
		layer_params[i].w.decay = 0.005;
		layer_params[i].w.learn_rate = 0.0005;
		layer_params[i].w.momentum = 0.9;
		layer_params[i].bias.decay = 0;
		layer_params[i].bias.learn_rate = 0.001;
		layer_params[i].bias.momentum = 0.9;
	}
	ccv_convnet_train_param_t train_params = {
		.max_epoch = 100,
		.mini_batch = 128,
		.layer_params = layer_params,
	};
	for (i = 0; i < 128; i++)
	{
		ccv_dense_matrix_t* image = 0;
		ccv_read(argv[1], &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
		ccv_dense_matrix_t* c = 0;
		ccv_slice(image, (ccv_matrix_t**)&c, CCV_32F, 0, 0, 128, 128);
		ccv_matrix_free(image);
		categorized.type = CCV_CATEGORIZED_DENSE_MATRIX;
		categorized.matrix = c;
	}
	cwc_bench_runtime(convnet, categorizeds, train_params);
	ccv_disable_cache();
	return 0;
}
