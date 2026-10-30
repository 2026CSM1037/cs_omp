#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <new>
using namespace std;

// Matrices are flattened 1D vectors: element (i, j) is at i * n + j

typedef void (*TransposeFunc)(const vector<double>&, vector<double>&, int);

void transpose_serial(const vector<double>& A, vector<double>& B, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[j * n + i] = A[i * n + j];
}

void transpose_outer(const vector<double>& A, vector<double>& B, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[j * n + i] = A[i * n + j];
}

void transpose_collapse(const vector<double>& A, vector<double>& B, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[j * n + i] = A[i * n + j];
}

// Bonus: work on small square tiles so the data stays in the cache
void transpose_blocked(const vector<double>& A, vector<double>& B, int n) {
    const int BS = 32;
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += BS)
        for (int jj = 0; jj < n; jj += BS)
            for (int i = ii; i < min(ii + BS, n); i++)
                for (int j = jj; j < min(jj + BS, n); j++)
                    B[j * n + i] = A[i * n + j];
}

double time_it(TransposeFunc f, const vector<double>& A, vector<double>& B, int n, int reps) {
    double start = omp_get_wtime();
    for (int r = 0; r < reps; r++)
        f(A, B, n);
    return (omp_get_wtime() - start) / reps * 1000.0;
}

int main(int argc, char* argv[]) {
    int max_n = 8192;
    if (argc > 1) max_n = atoi(argv[1]);

    int sizes[] = {1024, 4096, 8192};
    int thread_counts[] = {1, 2, 4, 8};

    printf("Matrices are flattened 1D vectors. Threads set with omp_set_num_threads().\n");
    printf("Times are averages over several repetitions, in milliseconds.\n\n");

    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        if (n > max_n) continue;
        int reps = (int)(50000000LL / ((long long)n * n));
        if (reps < 1) reps = 1;

        try {
            size_t total = (size_t)n * n;
            vector<double> A(total), ref(total), B(total);
            for (size_t k = 0; k < total; k++) A[k] = (double)k;

            transpose_serial(A, ref, n);
            double serial_ms = time_it(transpose_serial, A, ref, n, reps);

            printf("N = %d   (serial = %.3f ms)\n", n, serial_ms);
            printf("Threads | Outer(ms) Collapse(ms) Blocked(ms) | Spd(outer) Spd(collapse) Spd(blocked) | Correct\n");

            for (int t = 0; t < 4; t++) {
                omp_set_num_threads(thread_counts[t]);

                TransposeFunc funcs[3] = {transpose_outer, transpose_collapse, transpose_blocked};
                double ms[3];
                bool ok = true;
                for (int v = 0; v < 3; v++) {
                    fill(B.begin(), B.end(), -1.0);
                    funcs[v](A, B, n);
                    if (B != ref) ok = false;
                    ms[v] = time_it(funcs[v], A, B, n, reps);
                }

                printf("%7d | %9.3f %12.3f %11.3f | %10.2f %13.2f %12.2f | %s\n",
                       thread_counts[t], ms[0], ms[1], ms[2],
                       serial_ms / ms[0], serial_ms / ms[1], serial_ms / ms[2],
                       ok ? "YES" : "NO");
            }
            printf("\n");
        } catch (bad_alloc&) {
            printf("Not enough memory for N = %d, skipping.\n\n", n);
        }
    }
    return 0;
}
