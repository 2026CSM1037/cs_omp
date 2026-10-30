#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <new>
#include <bits/stdc++.h>
using namespace std;

void vec_add_serial(const vector<double>& a, const vector<double>& b, vector<double>& c, long long n) {
    for (long long i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vec_add_parallel(const vector<double>& a, const vector<double>& b, vector<double>& c, long long n) {
    #pragma omp parallel for
    for (long long i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

int main(int argc, char* argv[]) {
    long long max_n = 100000000;
    if (argc > 1) max_n = atoll(argv[1]);

    long long sizes[] = {100000, 1000000, 10000000, 100000000};
    int thread_counts[] = {1, 2, 4, 8};

    printf("Threads are set with omp_set_num_threads() in the code.\n");
    printf("Cores/threads visible to OpenMP: %d\n\n", omp_get_max_threads());

    for (int s = 0; s < 4; s++) {
        long long n = sizes[s];
        if (n > max_n) continue;

        try {
            vector<double> a(n), b(n), c_serial(n), c_parallel(n);

            for (long long i = 0; i < n; i++) {
                a[i] = (double)i;
                b[i] = (double)(2 * i);
            }

            double t0 = omp_get_wtime();
            vec_add_serial(a, b, c_serial, n);
            double t1 = omp_get_wtime();
            double serial_time = t1 - t0;

            printf("N = %lld   (serial time = %.6f s)\n", n, serial_time);
            printf("Threads   Parallel(s)   Speedup   Match\n");

            for (int t = 0; t < 4; t++) {
                int threads = thread_counts[t];
                omp_set_num_threads(threads);

                for (long long i = 0; i < n; i++) 
                    c_parallel[i] = -1.0;

                double start = omp_get_wtime();
                vec_add_parallel(a, b, c_parallel, n);
                double end = omp_get_wtime();
                double parallel_time = end - start;

                bool same = true;
                for (long long i = 0; i < n; i++) {
                    if (c_serial[i] != c_parallel[i]) {
                        same = false;
                        break;
                    }
                }
                printf("%7d   %12.6f   %7.2f   %s\n", threads, parallel_time,
                       serial_time / parallel_time, same ? "YES" : "NO");
            }
            printf("\n");
        } catch (bad_alloc&) {
            printf("Not enough memory for N = %lld, skipping.\n\n", n);
        }
    }
    return 0;
}
