__kernel void reduction(__global int* numbers, __global int* sum, int capacity, __local int * temp) {

	int i = get_global_id(0);
	int temp_i = get_global_id(0);


	// Initial Array Temp
	// 유효한 인덱스만 초기화
	temp[temp_i] = (i < capacity) ? numbers[i] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int offset = get_local_size(0); offset >= 1; offset = offset >> 1) {


		if (offset > temp_i) {
			temp[temp_i] += numbers[temp_i + offset];
		}
	
		barrier(CLK_LOCAL_MEM_FENCE);

	}

	if (temp_i == 0) {
		sum[get_group_id(0)] = temp[0];
	}


}