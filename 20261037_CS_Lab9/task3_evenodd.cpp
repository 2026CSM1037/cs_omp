#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <new>
using namespace std;

void count_serial(const vector<int>& a, long long n, long long& even_count, long long& odd_count) {
    long long even = 0, odd = 0;
    for (long long i = 0; i < n; i++) {
        if (a[i] % 2 == 0)
            even++;
        else
            odd++;
    }
    even_count = even;
    odd_count = odd;
}

void count_reduction(const vector<int>& a, long long n, long long& even_count, long long& odd_count) {
    long long even = 0, odd = 0;
    #pragma omp parallel for reduction(+:even, odd)
    for (long long i = 0; i < n; i++) {
        if (a[i] % 2 == 0)
            even++;
        else
            odd++;
    }
    even_count = even;
    odd_count = odd;
}

void count_atomic(const vector<int>& a, long long n, long long& even_count, long long& odd_count) {
    long long even = 0, odd = 0;
    #pragma omp parallel for
    for (long long i = 0; i < n; i++) {
        if (a[i] % 2 == 0) {
            #pragma omp atomic
            even++;
        } else {
            #pragma omp atomic
            odd++;
        }
    }
    even_count = even;
    odd_count = odd;
}

int main(int argc, char* argv[]) {
    long long max_n = 100000000;
    if (argc > 1) max_n = atoll(argv[1]);

    long long sizes[] = {100000, 1000000, 10000000, 100000000};
    int thread_counts[] = {1, 2, 4, 8};

    printf("Threads are set with omp_set_num_threads() in the code.\n");
    printf("Times are averages over several repetitions, in milliseconds.\n\n");

    for (int s = 0; s < 4; s++) {
        long long n = sizes[s];
        if (n > max_n) continue;
        int reps = (int)(100000000LL / n);
        if (reps < 1) reps = 1;

        try {
            vector<int> a(n);
            mt19937 gen(12345);
            for (long long i = 0; i < n; i++)
                a[i] = (int)(gen() % 1000000000);

            long long serial_even = 0, serial_odd = 0;
            double start = omp_get_wtime();
            for (int r = 0; r < reps; r++)
                count_serial(a, n, serial_even, serial_odd);
            double serial_ms = (omp_get_wtime() - start) / reps * 1000.0;

            printf("N = %lld   (serial = %.3f ms, even = %lld, odd = %lld, even + odd = N: %s)\n",
                   n, serial_ms, serial_even, serial_odd,
                   serial_even + serial_odd == n ? "YES" : "NO");
            printf("Threads | Reduction(ms) Speedup Efficiency | Atomic(ms) Speedup Efficiency | Correct\n");

            for (int t = 0; t < 4; t++) {
                int threads = thread_counts[t];
                omp_set_num_threads(threads);
                long long even = 0, odd = 0;
                bool ok = true;

                start = omp_get_wtime();
                for (int r = 0; r < reps; r++)
                    count_reduction(a, n, even, odd);
                double red_ms = (omp_get_wtime() - start) / reps * 1000.0;
                if (even != serial_even || odd != serial_odd || even + odd != n) ok = false;

                start = omp_get_wtime();
                for (int r = 0; r < reps; r++)
                    count_atomic(a, n, even, odd);
                double atom_ms = (omp_get_wtime() - start) / reps * 1000.0;
                if (even != serial_even || odd != serial_odd || even + odd != n) ok = false;

                printf("%7d | %13.3f %7.2f %10.2f | %10.3f %7.2f %10.2f | %s\n", threads,
                       red_ms, serial_ms / red_ms, serial_ms / red_ms / threads,
                       atom_ms, serial_ms / atom_ms, serial_ms / atom_ms / threads,
                       ok ? "YES" : "NO");
            }
            printf("\n");
        } catch (bad_alloc&) {
            printf("Not enough memory for N = %lld, skipping.\n\n", n);
        }
    }
    return 0;
}
