#include <omp.h>
#include <cstdio>
#include <vector>
using namespace std;

// Matrices are stored as flattened 1D vectors: element (i, j) is at i * n + j

void add_serial(const vector<double>& A, const vector<double>& B, vector<double>& C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i * n + j] = A[i * n + j] + B[i * n + j];
}

void add_outer(const vector<double>& A, const vector<double>& B, vector<double>& C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i * n + j] = A[i * n + j] + B[i * n + j];
}

void add_collapse(const vector<double>& A, const vector<double>& B, vector<double>& C, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i * n + j] = A[i * n + j] + B[i * n + j];
}

int main() {
    int sizes[] = {8, 4096};
    int thread_counts[] = {1, 2, 4, 8};

    printf("Matrices are flattened 1D vectors. Threads set with omp_set_num_threads().\n");
    printf("Each time is the average over several repetitions, in microseconds.\n\n");

    for (int s = 0; s < 2; s++) {
        int n = sizes[s];
        int reps = (n <= 64) ? 20000 : 5;
        size_t total = (size_t)n * n;

        vector<double> A(total), B(total), C_serial(total), C_outer(total), C_collapse(total);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = i + j;
                B[i * n + j] = i - j;
            }
        }

        add_serial(A, B, C_serial, n);   // warm-up and reference result

        double t0 = omp_get_wtime();
        for (int r = 0; r < reps; r++) add_serial(A, B, C_serial, n);
        double serial_time = (omp_get_wtime() - t0) / reps * 1e6;

        printf("N = %d   (serial = %.2f us)\n", n, serial_time);
        printf("Threads   Outer(us)   Collapse(us)   Speedup(outer)   Speedup(collapse)   Correct\n");

        for (int t = 0; t < 4; t++) {
            omp_set_num_threads(thread_counts[t]);

            for (size_t k = 0; k < total; k++) { C_outer[k] = -1; C_collapse[k] = -1; }
            add_outer(A, B, C_outer, n);
            add_collapse(A, B, C_collapse, n);
            bool ok = (C_outer == C_serial) && (C_collapse == C_serial);

            t0 = omp_get_wtime();
            for (int r = 0; r < reps; r++) add_outer(A, B, C_outer, n);
            double outer_time = (omp_get_wtime() - t0) / reps * 1e6;

            t0 = omp_get_wtime();
            for (int r = 0; r < reps; r++) add_collapse(A, B, C_collapse, n);
            double collapse_time = (omp_get_wtime() - t0) / reps * 1e6;

            printf("%7d   %9.2f   %12.2f   %14.2f   %17.2f   %s\n", thread_counts[t],
                   outer_time, collapse_time, serial_time / outer_time,
                   serial_time / collapse_time, ok ? "YES" : "NO");
        }
        printf("\n");
    }
    return 0;
}
