__kernel void image_rotate (__global float * input, __global float * output, int W, int H, char * _theta) {
	
	int dest_x = get_global_id(0);
	int dest_y = get_global_id(1);

	float x0 = W / 2.0f;
	float y0 = H / 2.0f;
	
	const float theta = atof(_theta) * M_PI / 180;
	const float sin_theta = sinf(theta);
	const float cos_theta = cosf(theta);
	
	float xOff = dest_x - x0;
	float yOff = dest_y - y0;

	int src_x = (int)(xOff * cos_theta + yOff * sin_theta + x0);
	int src_y = (int)(yOff * cos_theta - xOff * sin_theta + y0);

	if ((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H)) {
		output[dest_y * W + dest_x] = input[src_y * W + src_x];
	} else {
		output[dest_y * W + dest_x] = 0.0f;
	}

}