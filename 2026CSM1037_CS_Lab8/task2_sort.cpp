#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

void bubble_sort_serial(vector<int>& a, int n) {
    for (int pass = 0; pass < n - 1; pass++)
        for (int i = 0; i < n - 1 - pass; i++)
            if (a[i] > a[i + 1])
                swap(a[i], a[i + 1]);
}

// Same algorithm as the parallel version below, but with no OpenMP
void odd_even_sort_serial(vector<int>& a, int n) {
    for (int phase = 0; phase < n; phase++) {
        int start = phase % 2;
        for (int i = start; i < n - 1; i += 2)
            if (a[i] > a[i + 1])
                swap(a[i], a[i + 1]);
    }
}

// Parallel bubble sort = odd-even transposition sort
void odd_even_sort_parallel(vector<int>& a, int n) {
    #pragma omp parallel
    {
        for (int phase = 0; phase < n; phase++) {
            int start = phase % 2;
            #pragma omp for
            for (int i = start; i < n - 1; i += 2)
                if (a[i] > a[i + 1])
                    swap(a[i], a[i + 1]);
            // the end of "omp for" has a barrier: nobody starts the next phase early
        }
    }
}

int main(int argc, char* argv[]) {
    int max_n = 40000;
    if (argc > 1) max_n = atoi(argv[1]);

    int sizes[] = {10000, 20000, 40000};
    int thread_counts[] = {1, 2, 4, 8};

    printf("Threads are set with omp_set_num_threads() in the code.\n");
    printf("Parallel version = odd-even transposition sort (a parallel bubble sort).\n\n");

    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        if (n > max_n) continue;

        vector<int> original(n);
        mt19937 gen(12345);
        for (int i = 0; i < n; i++)
            original[i] = (int)(gen() % 1000000);

        vector<int> reference = original;
        double start = omp_get_wtime();
        sort(reference.begin(), reference.end());
        double std_time = omp_get_wtime() - start;

        vector<int> work = original;
        start = omp_get_wtime();
        bubble_sort_serial(work, n);
        double serial_time = omp_get_wtime() - start;
        bool serial_ok = (work == reference);

        work = original;
        start = omp_get_wtime();
        odd_even_sort_serial(work, n);
        double oe_serial_time = omp_get_wtime() - start;
        bool oe_ok = (work == reference);

        printf("N = %d\n", n);
        printf("Serial bubble sort:        %.4f s (correct: %s)\n", serial_time, serial_ok ? "YES" : "NO");
        printf("Serial odd-even sort:      %.4f s (correct: %s)\n", oe_serial_time, oe_ok ? "YES" : "NO");
        printf("std::sort (for reference): %.4f s\n", std_time);
        printf("Threads   Parallel(s)   Speedup vs bubble   Speedup vs serial odd-even   Efficiency   Correct\n");

        for (int t = 0; t < 4; t++) {
            int threads = thread_counts[t];
            omp_set_num_threads(threads);

            work = original;
            start = omp_get_wtime();
            odd_even_sort_parallel(work, n);
            double par_time = omp_get_wtime() - start;

            double speedup_oe = oe_serial_time / par_time;
            printf("%7d   %11.4f   %17.2f   %26.2f   %10.2f   %s\n", threads, par_time,
                   serial_time / par_time, speedup_oe, speedup_oe / threads,
                   work == reference ? "YES" : "NO");
        }
        printf("\n");
    }
    return 0;
}
