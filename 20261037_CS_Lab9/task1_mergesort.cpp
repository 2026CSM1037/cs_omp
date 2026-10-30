#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

void merge(int* a, int* tmp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (a[i] <= a[j]) tmp[k++] = a[i++];
        else              tmp[k++] = a[j++];
    }
    while (i <= mid)   tmp[k++] = a[i++];
    while (j <= right) tmp[k++] = a[j++];
    for (k = left; k <= right; k++)
        a[k] = tmp[k];
}

void merge_sort_serial(int* a, int* tmp, int left, int right) {
    if (left >= right)
        return;
    int mid = left + (right - left) / 2;
    merge_sort_serial(a, tmp, left, mid);
    merge_sort_serial(a, tmp, mid + 1, right);
    merge(a, tmp, left, mid, right);
}

void merge_sort_parallel(int* a, int* tmp, int left, int right, int cutoff) {
    if (left >= right)
        return;
    if (right - left < cutoff) {
        merge_sort_serial(a, tmp, left, right);
        return;
    }
    int mid = left + (right - left) / 2;

    #pragma omp task shared(a, tmp) firstprivate(left, mid, cutoff)
    merge_sort_parallel(a, tmp, left, mid, cutoff);

    #pragma omp task shared(a, tmp) firstprivate(mid, right, cutoff)
    merge_sort_parallel(a, tmp, mid + 1, right, cutoff);

    #pragma omp taskwait
    merge(a, tmp, left, mid, right);
}

void parallel_merge_sort(vector<int>& a, vector<int>& tmp, int n, int cutoff) {
    #pragma omp parallel
    {
        #pragma omp single
        merge_sort_parallel(a.data(), tmp.data(), 0, n - 1, cutoff);
    }
}

int main(int argc, char* argv[]) {
    int max_n = 10000000;
    if (argc > 1) max_n = atoi(argv[1]);

    int sizes[] = {100000, 1000000, 10000000};
    int thread_counts[] = {1, 2, 4, 8};
    const int CUTOFF = 2048;

    printf("Threads are set with omp_set_num_threads() in the code.\n");
    printf("Task cutoff = %d elements. Times are averages in milliseconds.\n\n", CUTOFF);

    // Part 1: main experiment
    printf("PART 1: serial vs parallel merge sort\n\n");
    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        if (n > max_n) continue;
        int reps = 5000000 / n;
        if (reps < 1) reps = 1;

        vector<int> original(n), work(n), tmp(n), serial_result;
        mt19937 gen(12345);
        for (int i = 0; i < n; i++) original[i] = (int)(gen() % 1000000);

        double total = 0;
        for (int r = 0; r < reps; r++) {
            work = original;
            double start = omp_get_wtime();
            merge_sort_serial(work.data(), tmp.data(), 0, n - 1);
            total += omp_get_wtime() - start;
        }
        double serial_ms = total / reps * 1000.0;
        serial_result = work;
        bool serial_ok = is_sorted(work.begin(), work.end());

        printf("N = %d   (serial = %.3f ms, sorted correctly: %s)\n", n, serial_ms,
               serial_ok ? "YES" : "NO");
        printf("Threads   Parallel(ms)   Speedup   Efficiency   Sorted+Matches serial\n");

        for (int t = 0; t < 4; t++) {
            int threads = thread_counts[t];
            omp_set_num_threads(threads);

            total = 0;
            bool ok = true;
            for (int r = 0; r < reps; r++) {
                work = original;
                double start = omp_get_wtime();
                parallel_merge_sort(work, tmp, n, CUTOFF);
                total += omp_get_wtime() - start;
                if (!is_sorted(work.begin(), work.end()) || work != serial_result) ok = false;
            }
            double par_ms = total / reps * 1000.0;
            double speedup = serial_ms / par_ms;
            printf("%7d   %12.3f   %7.2f   %10.2f   %s\n", threads, par_ms, speedup,
                   speedup / threads, ok ? "YES" : "NO");
        }
        printf("\n");
    }

    // Part 2: effect of the cutoff
    if (max_n >= 1000000) {
        int n = 1000000;
        int reps = 3;
        int cutoffs[] = {4, 32, 256, 2048, 16384, 131072, 500000};

        vector<int> original(n), work(n), tmp(n);
        mt19937 gen(12345);
        for (int i = 0; i < n; i++) original[i] = (int)(gen() % 1000000);

        double total = 0;
        for (int r = 0; r < reps; r++) {
            work = original;
            double start = omp_get_wtime();
            merge_sort_serial(work.data(), tmp.data(), 0, n - 1);
            total += omp_get_wtime() - start;
        }
        double serial_ms = total / reps * 1000.0;
        vector<int> serial_result = work;

        printf("PART 2: effect of the cutoff (N = %d, serial = %.3f ms)\n", n, serial_ms);
        printf("Each cell is: parallel time in ms (speedup)\n");
        printf("Cutoff    1 thread          2 threads         4 threads         8 threads         Correct\n");

        for (int c = 0; c < 7; c++) {
            printf("%6d", cutoffs[c]);
            bool ok = true;
            for (int t = 0; t < 4; t++) {
                omp_set_num_threads(thread_counts[t]);
                total = 0;
                for (int r = 0; r < reps; r++) {
                    work = original;
                    double start = omp_get_wtime();
                    parallel_merge_sort(work, tmp, n, cutoffs[c]);
                    total += omp_get_wtime() - start;
                    if (work != serial_result) ok = false;
                }
                double ms = total / reps * 1000.0;
                printf("   %9.3f (%5.2f)", ms, serial_ms / ms);
            }
            printf("   %s\n", ok ? "YES" : "NO");
        }
        printf("\n");
    }
    return 0;
}
