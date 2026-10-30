#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <new>
using namespace std;

const int BINS = 10;

void hist_serial(const vector<int>& a, long long n, long long* hist) {
    for (int b = 0; b < BINS; b++) hist[b] = 0;
    for (long long i = 0; i < n; i++)
        hist[a[i]]++;
}

void hist_naive_broken(const vector<int>& a, long long n, long long* hist) {
    for (int b = 0; b < BINS; b++) hist[b] = 0;
    #pragma omp parallel for
    for (long long i = 0; i < n; i++)
        hist[a[i]]++;   // data race on purpose
}

void hist_atomic(const vector<int>& a, long long n, long long* hist) {
    for (int b = 0; b < BINS; b++) hist[b] = 0;
    #pragma omp parallel for
    for (long long i = 0; i < n; i++) {
        #pragma omp atomic
        hist[a[i]]++;
    }
}

void hist_critical(const vector<int>& a, long long n, long long* hist) {
    for (int b = 0; b < BINS; b++) hist[b] = 0;
    #pragma omp parallel for
    for (long long i = 0; i < n; i++) {
        #pragma omp critical
        hist[a[i]]++;
    }
}

void hist_local_copy(const vector<int>& a, long long n, long long* hist) {
    for (int b = 0; b < BINS; b++) hist[b] = 0;
    #pragma omp parallel
    {
        long long local[BINS] = {0};

        #pragma omp for
        for (long long i = 0; i < n; i++)
            local[a[i]]++;

        #pragma omp critical
        for (int b = 0; b < BINS; b++)
            hist[b] += local[b];
    }
}

void hist_reduction(const vector<int>& a, long long n, long long* hist) {
    for (int b = 0; b < BINS; b++) hist[b] = 0;
    #pragma omp parallel for reduction(+:hist[0:BINS])
    for (long long i = 0; i < n; i++)
        hist[a[i]]++;
}

bool same_hist(const long long* x, const long long* y) {
    for (int b = 0; b < BINS; b++)
        if (x[b] != y[b]) return false;
    return true;
}

void print_hist(const char* label, const long long* h) {
    long long total = 0;
    printf("%-14s", label);
    for (int b = 0; b < BINS; b++) {
        printf("%10lld", h[b]);
        total += h[b];
    }
    printf("   total = %lld\n", total);
}

int main(int argc, char* argv[]) {
    long long max_n = 100000000;
    if (argc > 1) max_n = atoll(argv[1]);

    int thread_counts[] = {1, 2, 4, 8};
    long long sizes[] = {10000000, 100000000};

    printf("Threads are set with omp_set_num_threads() in the code.\n");
    printf("WARNING: critical and atomic versions are slow for big N.\n\n");

    // Part 1: race condition demo
    {
        long long n = 10000000;
        if (n > max_n) n = max_n;
        vector<int> a(n);
        mt19937 gen(12345);
        for (long long i = 0; i < n; i++) a[i] = (int)(gen() % BINS);

        long long correct[BINS], naive[BINS];
        hist_serial(a, n, correct);
        omp_set_num_threads(4);

        printf("PART 1: race condition demo (N = %lld, 4 threads)\n", n);
        printf("%-14s", "Value");
        for (int b = 0; b < BINS; b++) printf("%10d", b);
        printf("\n");
        print_hist("Serial", correct);
        for (int run = 1; run <= 3; run++) {
            hist_naive_broken(a, n, naive);
            char label[32];
            snprintf(label, sizeof(label), "Naive run %d", run);
            print_hist(label, naive);
        }
        printf("\n");
    }

    // Part 2: timing
    for (int s = 0; s < 2; s++) {
        long long n = sizes[s];
        if (n > max_n) continue;

        try {
            vector<int> a(n);
            mt19937 gen(12345);
            for (long long i = 0; i < n; i++) a[i] = (int)(gen() % BINS);

            long long ref[BINS], h[BINS];
            double start = omp_get_wtime();
            hist_serial(a, n, ref);
            double serial_time = omp_get_wtime() - start;

            printf("PART 2: N = %lld   (serial time = %.6f s)\n", n, serial_time);
            printf("Time in seconds\n");
            printf("Threads    Atomic   Critical  LocalCopy  Reduction   Correct\n");

            double speed[4][4];
            for (int t = 0; t < 4; t++) {
                omp_set_num_threads(thread_counts[t]);
                bool ok = true;
                double times[4];

                start = omp_get_wtime();
                hist_atomic(a, n, h);
                times[0] = omp_get_wtime() - start;
                if (!same_hist(ref, h)) ok = false;

                start = omp_get_wtime();
                hist_critical(a, n, h);
                times[1] = omp_get_wtime() - start;
                if (!same_hist(ref, h)) ok = false;

                start = omp_get_wtime();
                hist_local_copy(a, n, h);
                times[2] = omp_get_wtime() - start;
                if (!same_hist(ref, h)) ok = false;

                start = omp_get_wtime();
                hist_reduction(a, n, h);
                times[3] = omp_get_wtime() - start;
                if (!same_hist(ref, h)) ok = false;

                printf("%7d %9.4f %10.4f %10.4f %10.4f   %s\n", thread_counts[t],
                       times[0], times[1], times[2], times[3], ok ? "YES" : "NO");
                for (int v = 0; v < 4; v++) speed[t][v] = serial_time / times[v];
            }

            printf("Speedup (serial time / parallel time)\n");
            printf("Threads    Atomic   Critical  LocalCopy  Reduction\n");
            for (int t = 0; t < 4; t++)
                printf("%7d %9.2f %10.2f %10.2f %10.2f\n", thread_counts[t],
                       speed[t][0], speed[t][1], speed[t][2], speed[t][3]);
            printf("\n");
        } catch (bad_alloc&) {
            printf("Not enough memory for N = %lld, skipping.\n\n", n);
        }
    }
    return 0;
}
