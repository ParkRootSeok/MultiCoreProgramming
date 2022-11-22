__kernel void integral (__global float * sum, int N) {
	
	int i = get_global_id(0);
	float dx = (1.0 / (double)N);
	float x = ((double)i * dx);

	sum += ((3 * x * x) + (2 * x) + 1) * dx;

}
