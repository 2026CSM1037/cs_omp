#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <new>
using namespace std;

long long sum_serial(const vector<int>& a, long long n) {
    long long sum = 0;
    for (long long i = 0; i < n; i++)
        sum += a[i];
    return sum;
}

long long sum_naive_broken(const vector<int>& a, long long n) {
    long long sum = 0;
    #pragma omp parallel for
    for (long long i = 0; i < n; i++)
        sum += a[i];   // data race on purpose
    return sum;
}

long long sum_reduction(const vector<int>& a, long long n) {
    long long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++)
        sum += a[i];
    return sum;
}

long long sum_critical(const vector<int>& a, long long n) {
    long long sum = 0;
    #pragma omp parallel for
    for (long long i = 0; i < n; i++) {
        #pragma omp critical
        sum += a[i];
    }
    return sum;
}

int main(int argc, char* argv[]) {
    long long max_n = 100000000;
    if (argc > 1) max_n = atoll(argv[1]);

    int thread_counts[] = {1, 2, 4, 8};
    printf("Threads are set with omp_set_num_threads() in the code.\n\n");

    // Part 1: show the data race
    {
        long long n = 10000000;
        if (n > max_n) n = max_n;
        vector<int> a(n);
        for (long long i = 0; i < n; i++) a[i] = (int)(i % 10) + 1;

        long long expected = sum_serial(a, n);
        omp_set_num_threads(4);

        printf("PART 1: race condition demo (N = %lld, 4 threads)\n", n);
        printf("Correct answer (serial): %lld\n", expected);
        printf("Run   Naive(no reduction)   Reduction   Naive correct?\n");
        for (int run = 1; run <= 5; run++) {
            long long naive = sum_naive_broken(a, n);
            long long red = sum_reduction(a, n);
            printf("%3d   %19lld   %9lld   %s\n", run, naive, red,
                   naive == expected ? "yes" : "NO");
        }
        printf("\n");
    }

    // Part 2: reduction timing
    printf("PART 2: reduction timing\n\n");
    long long sizes[] = {10000000, 100000000};
    for (int s = 0; s < 2; s++) {
        long long n = sizes[s];
        if (n > max_n) continue;
        try {
            vector<int> a(n);
            for (long long i = 0; i < n; i++) a[i] = (int)(i % 10) + 1;

            double t0 = omp_get_wtime();
            long long expected = sum_serial(a, n);
            double serial_time = omp_get_wtime() - t0;

            printf("N = %lld   (serial time = %.6f s)\n", n, serial_time);
            printf("Threads   Reduction(s)   Speedup   Correct\n");
            for (int t = 0; t < 4; t++) {
                omp_set_num_threads(thread_counts[t]);
                double start = omp_get_wtime();
                long long result = sum_reduction(a, n);
                double time = omp_get_wtime() - start;
                printf("%7d   %13.6f   %7.2f   %s\n", thread_counts[t], time,
                       serial_time / time, result == expected ? "YES" : "NO");
            }
            printf("\n");
        } catch (bad_alloc&) {
            printf("Not enough memory for N = %lld, skipping.\n\n", n);
        }
    }

    // Part 3: critical timing (only N = 10^7, it is very slow)
    {
        long long n = 10000000;
        if (n <= max_n) {
            vector<int> a(n);
            for (long long i = 0; i < n; i++) a[i] = (int)(i % 10) + 1;
            long long expected = sum_serial(a, n);

            printf("PART 3: critical vs reduction (N = %lld)\n", n);
            printf("Threads   Critical(s)   Reduction(s)   Critical correct\n");
            for (int t = 0; t < 4; t++) {
                omp_set_num_threads(thread_counts[t]);

                double start = omp_get_wtime();
                long long r1 = sum_critical(a, n);
                double crit_time = omp_get_wtime() - start;

                start = omp_get_wtime();
                sum_reduction(a, n);
                double red_time = omp_get_wtime() - start;

                printf("%7d   %11.6f   %12.6f   %s\n", thread_counts[t], crit_time,
                       red_time, r1 == expected ? "YES" : "NO");
            }
            printf("\n");
        }
    }
    return 0;
}
