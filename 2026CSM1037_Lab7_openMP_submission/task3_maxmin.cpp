#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <new>
using namespace std;

// want_max = true  -> looking for the maximum
// want_max = false -> looking for the minimum
inline bool is_better(int x, int y, bool want_max) {
    return want_max ? (x > y) : (x < y);
}

// If a value appears many times, every version returns the SMALLEST index.

int max_value_reduction(const vector<int>& a, long long n) {
    int max_val = a[0];
    #pragma omp parallel for reduction(max:max_val)
    for (long long i = 0; i < n; i++)
        if (a[i] > max_val) max_val = a[i];
    return max_val;
}

int min_value_reduction(const vector<int>& a, long long n) {
    int min_val = a[0];
    #pragma omp parallel for reduction(min:min_val)
    for (long long i = 0; i < n; i++)
        if (a[i] < min_val) min_val = a[i];
    return min_val;
}

void find_idx_serial(const vector<int>& a, long long n, bool want_max,
                     int& out_val, long long& out_idx) {
    int best_val = a[0];
    long long best_idx = 0;
    for (long long i = 1; i < n; i++) {
        if (is_better(a[i], best_val, want_max)) {
            best_val = a[i];
            best_idx = i;
        }
    }
    out_val = best_val;
    out_idx = best_idx;
}

void find_idx_critical(const vector<int>& a, long long n, bool want_max,
                       int& out_val, long long& out_idx) {
    int best_val = a[0];
    long long best_idx = 0;
    #pragma omp parallel for
    for (long long i = 1; i < n; i++) {
        #pragma omp critical
        {
            if (is_better(a[i], best_val, want_max) ||
                (a[i] == best_val && i < best_idx)) {
                best_val = a[i];
                best_idx = i;
            }
        }
    }
    out_val = best_val;
    out_idx = best_idx;
}

void find_idx_manual(const vector<int>& a, long long n, bool want_max,
                     int& out_val, long long& out_idx) {
    int max_threads = omp_get_max_threads();
    vector<int> local_val(max_threads, 0);
    vector<long long> local_idx(max_threads, -1);

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int num = omp_get_num_threads();
        long long chunk = (n + num - 1) / num;
        long long start = id * chunk;
        long long end = start + chunk;
        if (end > n) end = n;

        if (start < end) {
            int best_val = a[start];
            long long best_idx = start;
            for (long long i = start + 1; i < end; i++) {
                if (is_better(a[i], best_val, want_max)) {
                    best_val = a[i];
                    best_idx = i;
                }
            }
            local_val[id] = best_val;
            local_idx[id] = best_idx;
        }
    }

    // serial pass over the per-thread results
    int best_val = 0;
    long long best_idx = -1;
    for (int t = 0; t < max_threads; t++) {
        if (local_idx[t] == -1) continue;
        if (best_idx == -1 || is_better(local_val[t], best_val, want_max)) {
            best_val = local_val[t];
            best_idx = local_idx[t];
        }
    }
    out_val = best_val;
    out_idx = best_idx;
}

int main(int argc, char* argv[]) {
    long long max_n = 100000000;
    if (argc > 1) max_n = atoll(argv[1]);

    long long sizes[] = {10000000, 100000000};
    int thread_counts[] = {1, 2, 4, 8};

    printf("Threads are set with omp_set_num_threads() in the code.\n");
    printf("WARNING: the critical version is very slow for big N.\n\n");

    for (int s = 0; s < 2; s++) {
        long long n = sizes[s];
        if (n > max_n) continue;

        try {
            vector<int> a(n);
            mt19937 gen(12345);
            for (long long i = 0; i < n; i++)
                a[i] = (int)(gen() % 1000000000);

            int smax, smin;
            long long smax_i, smin_i;

            double t0 = omp_get_wtime();
            find_idx_serial(a, n, true, smax, smax_i);
            double serial_max_time = omp_get_wtime() - t0;

            t0 = omp_get_wtime();
            find_idx_serial(a, n, false, smin, smin_i);
            double serial_min_time = omp_get_wtime() - t0;

            printf("N = %lld\n", n);
            printf("Serial: max = %d at index %lld (%.6f s), min = %d at index %lld (%.6f s)\n",
                   smax, smax_i, serial_max_time, smin, smin_i, serial_min_time);
            printf("Times in seconds\n");
            printf("Threads | MaxVal(red) MaxIdx(crit) MaxIdx(man) | MinVal(red) MinIdx(crit) MinIdx(man) | Correct\n");

            for (int t = 0; t < 4; t++) {
                omp_set_num_threads(thread_counts[t]);
                int v;
                long long idx;
                bool ok = true;

                double start = omp_get_wtime();
                v = max_value_reduction(a, n);
                double t_max_red = omp_get_wtime() - start;
                if (v != smax) ok = false;

                start = omp_get_wtime();
                find_idx_critical(a, n, true, v, idx);
                double t_max_crit = omp_get_wtime() - start;
                if (v != smax || idx != smax_i) ok = false;

                start = omp_get_wtime();
                find_idx_manual(a, n, true, v, idx);
                double t_max_man = omp_get_wtime() - start;
                if (v != smax || idx != smax_i) ok = false;

                start = omp_get_wtime();
                v = min_value_reduction(a, n);
                double t_min_red = omp_get_wtime() - start;
                if (v != smin) ok = false;

                start = omp_get_wtime();
                find_idx_critical(a, n, false, v, idx);
                double t_min_crit = omp_get_wtime() - start;
                if (v != smin || idx != smin_i) ok = false;

                start = omp_get_wtime();
                find_idx_manual(a, n, false, v, idx);
                double t_min_man = omp_get_wtime() - start;
                if (v != smin || idx != smin_i) ok = false;

                printf("%7d | %11.6f %12.6f %11.6f | %11.6f %12.6f %11.6f | %s\n",
                       thread_counts[t], t_max_red, t_max_crit, t_max_man,
                       t_min_red, t_min_crit, t_min_man, ok ? "YES" : "NO");
            }
            printf("\n");
        } catch (bad_alloc&) {
            printf("Not enough memory for N = %lld, skipping.\n\n", n);
        }
    }
    return 0;
}
