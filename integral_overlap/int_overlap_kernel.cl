__kernel void integral (__global float * sum, int N) {
	
	int i = get_global_id(0);
	float dx = (1000.0 / (float)N);
	float x = ((float)i * dx);

	sum[i] = ((3 * x * x) + (2 * x) + 1) * dx;

}
