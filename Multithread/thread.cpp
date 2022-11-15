#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>

using namespace std;
using namespace std::chrono;

mutex mylock;
volatile int sum;


void thread_func(int thread_num) {

	volatile int local_sum = 0;
	
	for (auto i = 0; i < 50000000 / thread_num; i++) {
		local_sum += 2;
	}
	
	mylock.lock();
	sum += local_sum;
	mylock.unlock();

}

void thread_func_Atomic(int thread_num) {

	for (auto i = 0; i < 50000000 / thread_num; i++) {

		_asm lock add sum, 2;

	}

}

int main() {

	thread t[4];

	for (int i = 1; i < 5; i *= 2) {
		
		sum = 0;

		
		for (int j = 0; j < i; j++) t[j] = thread{ thread_func, i };
		
		auto start = high_resolution_clock::now();
		for (int j = 0; j < i; j++) t[j].join();
		auto end = high_resolution_clock::now() - start;

		cout << i << " threads ";
		cout << " sum = " << sum ;
		cout << " duration = " << duration_cast<milliseconds>(end).count() << " ms\n";
	}
	
}