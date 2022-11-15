__kernel void reduction (__global int * numbers, __global int * sum, int capacity, __local int * local_sum) {

	int i = get_global_id(0);
	int local_Id = get_global_id(0);
	int group_size = get_local_size(0);
	
	// Initial Array Temp
	// 유효한 인덱스만 초기화
	local_sum[local_Id] = (i < capacity) ? numbers[i] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int offset = group_size/2 ; offset > 0 ; offset >>= 1) {

		if (offset > local_Id) {
			local_sum[local_Id] += numbers[local_Id + offset];
		}
	
		barrier(CLK_LOCAL_MEM_FENCE);

	}

	if (local_Id == 0) {
		sum[get_group_id(0)] = local_sum[0];
	}


}