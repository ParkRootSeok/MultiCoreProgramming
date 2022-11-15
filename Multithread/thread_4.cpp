#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>

using namespace std;
using namespace std::chrono;

mutex mylock;
int sum;


void thread_func() {

	for (auto i = 0; i < 50000000 / 4; i++) {
		mylock.lock();
		sum += 2;
		mylock.unlock();
	}

}

void thread_func_Atomic() {

	for (auto i = 0; i < 50000000 / 4; i++) {

		_asm lock add sum, 2;

	}

}

int main() {

	thread t1 = thread{ thread_func };
	thread t2 = thread{ thread_func };
	thread t3 = thread{ thread_func };
	thread t4 = thread{ thread_func };

	auto start = high_resolution_clock::now();
	t1.join();
	t2.join();
	t3.join();
	t4.join();
	auto end = high_resolution_clock::now() - start;

	cout << " sum = " << sum;
	cout << " duration = " << duration_cast<microseconds>(end).count() << " ¥ìs\n";


}