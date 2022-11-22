__kernel void reduction(__global int* numbers, __global int* sum, __local int* local_sum, int capacity) {
	
	int i = get_global_id(0);
	int local_id = get_local_id(0);

	local_sum[local_id] = (i < capacity) ? numbers[i] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int offset = get_local_size(0) / 2; offset >= 1; offset >>= 1) {
		if (local_id < offset) local_sum[local_id] += local_sum[local_id + offset];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_id == 0) {
		sum[get_group_id(0)] = local_sum[0];
	}

}