#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <new>
using namespace std;

const int KEY = 2000000000;   // bigger than any random value, so it only exists where we put it

void search_serial(const vector<int>& a, long long n, int key, long long& index) {
    index = -1;
    for (long long i = 0; i < n; i++) {
        if (a[i] == key) {
            index = i;
            break;
        }
    }
}

void search_parallel(const vector<int>& a, long long n, int key, long long& index) {
    long long first = n;
    #pragma omp parallel for reduction(min:first)
    for (long long i = 0; i < n; i++) {
        if (a[i] == key && i < first)
            first = i;
    }
    index = (first == n) ? -1 : first;
}

int main(int argc, char* argv[]) {
    long long max_n = 100000000;
    if (argc > 1) max_n = atoll(argv[1]);

    long long sizes[] = {10000000, 100000000};
    int thread_counts[] = {1, 2, 4, 8};
    const char* case_names[] = {"key at 10% of array", "key in the middle", "key at the very end",
                                "key twice (50%, 75%)", "key not present"};

    printf("Threads are set with omp_set_num_threads() in the code.\n");
    printf("Times are averages over several repetitions, in milliseconds.\n\n");

    for (int s = 0; s < 2; s++) {
        long long n = sizes[s];
        if (n > max_n) continue;
        int reps = (n <= 10000000) ? 20 : 5;

        try {
            vector<int> a(n);
            mt19937 gen(12345);
            for (long long i = 0; i < n; i++)
                a[i] = (int)(gen() % 1000000000);

            printf("N = %lld\n", n);
            printf("%-22s %7s %11s %13s %8s %12s  %s\n", "Case", "Threads", "Serial(ms)",
                   "Parallel(ms)", "Speedup", "Index", "Match");

            for (int c = 0; c < 5; c++) {
                long long p1 = -1, p2 = -1;
                if (c == 0) p1 = n / 10;
                if (c == 1) p1 = n / 2;
                if (c == 2) p1 = n - 1;
                if (c == 3) { p1 = n / 2; p2 = 3 * n / 4; }

                int saved1 = 0, saved2 = 0;
                if (p1 >= 0) { saved1 = a[p1]; a[p1] = KEY; }
                if (p2 >= 0) { saved2 = a[p2]; a[p2] = KEY; }

                long long expected = -1;
                double start = omp_get_wtime();
                for (int r = 0; r < reps; r++)
                    search_serial(a, n, KEY, expected);
                double serial_ms = (omp_get_wtime() - start) / reps * 1000.0;

                for (int t = 0; t < 4; t++) {
                    omp_set_num_threads(thread_counts[t]);
                    long long found = -2;

                    start = omp_get_wtime();
                    for (int r = 0; r < reps; r++)
                        search_parallel(a, n, KEY, found);
                    double par_ms = (omp_get_wtime() - start) / reps * 1000.0;

                    printf("%-22s %7d %11.3f %13.3f %8.2f %12lld  %s\n", case_names[c],
                           thread_counts[t], serial_ms, par_ms, serial_ms / par_ms, found,
                           found == expected ? "YES" : "NO");
                }

                if (p1 >= 0) a[p1] = saved1;
                if (p2 >= 0) a[p2] = saved2;
            }
            printf("\n");
        } catch (bad_alloc&) {
            printf("Not enough memory for N = %lld, skipping.\n\n", n);
        }
    }
    return 0;
}
