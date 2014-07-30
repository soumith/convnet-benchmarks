#include "ccv.h"
#include <ctype.h>

void cwc_bench_runtime(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_convnet_train_param_t params);

int main(int argc, char** argv)
{
	ccv_enable_default_cache();
	assert(argc == 2);
	FILE *r = fopen(argv[1], "r");
	char* file = (char*)malloc(1024);
	ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_categorized_t), 64, 0);
	size_t len = 1024;
	ssize_t read;
	while ((read = getline(&file, &len, r)) != -1)
	{
		while(read > 1 && isspace(file[read - 1]))
			read--;
		file[read] = 0;
		ccv_file_info_t input;
		input.filename = (char*)ccmalloc(1024);
		strncpy(input.filename, file, 1024);
		ccv_categorized_t categorized = ccv_categorized(0, 0, &input);
		ccv_array_push(categorizeds, &categorized);
	}
	fclose(r);
	free(file);
	/* convnet-benchmarks parameters */
	ccv_convnet_layer_param_t params[] = {
		// first layer (convolutional => max pool => rnorm)
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
					.partition = 1,
				},
			},
		},
		// second layer (convolutional => max pool => rnorm)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 1,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 64,
					.cols = 64,
					.channels = 64,
					.partition = 1,
				},
			},
			.output = {
				.convolutional = {
					.count = 128,
					.strides = 1,
					.border = 0,
					.rows = 9,
					.cols = 9,
					.channels = 64,
					.partition = 1,
				},
			},
		},
		// third layer (convolutional)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 0,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 32,
					.cols = 32,
					.channels = 128,
					.partition = 1,
				},
			},
			.output = {
				.convolutional = {
					.count = 128,
					.strides = 1,
					.border = 0,
					.rows = 9,
					.cols = 9,
					.channels = 128,
					.partition = 1,
				},
			},
		},
		// fourth layer (convolutional)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 1,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 16,
					.cols = 16,
					.channels = 128,
					.partition = 1,
				},
			},
			.output = {
				.convolutional = {
					.count = 128,
					.strides = 1,
					.border = 0,
					.rows = 7,
					.cols = 7,
					.channels = 128,
					.partition = 1,
				},
			},
		},
		// fifth layer (convolutional => max pool)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 1,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 13,
					.cols = 13,
					.channels = 384,
					.partition = 1,
				},
			},
			.output = {
				.convolutional = {
					.count = 384,
					.strides = 1,
					.border = 0,
					.rows = 3,
					.cols = 3,
					.channels = 384,
					.partition = 1,
				},
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(1, ccv_size(128, 128), params, sizeof(params) / sizeof(ccv_convnet_layer_param_t));
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
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, i);
		ccv_dense_matrix_t* image = 0;
		ccv_read(categorized->file.filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
		ccv_dense_matrix_t* b = 0;
		if (image->rows > 128 && image->cols > 128)
			ccv_resample(image, &b, 0, ccv_max(128, (int)(image->rows * 128.0 / image->cols + 0.5)), ccv_max(128, (int)(image->cols * 128.0 / image->rows + 0.5)), CCV_INTER_AREA);
		else if (image->rows < 128 || image->cols < 128)
			ccv_resample(image, &b, 0, ccv_max(128, (int)(image->rows * 128.0 / image->cols + 0.5)), ccv_max(128, (int)(image->cols * 128.0 / image->rows + 0.5)), CCV_INTER_CUBIC);
		else
			b = image;
		if (b != image)
			ccv_matrix_free(image);
		ccv_dense_matrix_t* c = 0;
		ccv_slice(b, (ccv_matrix_t**)&c, CCV_32F, 0, 0, 128, 128);
		ccv_matrix_free(b);
		categorized->type = CCV_CATEGORIZED_DENSE_MATRIX;
		categorized->matrix = c;
	}
	cwc_bench_runtime(convnet, categorizeds, train_params);
	ccv_disable_cache();
	return 0;
}
