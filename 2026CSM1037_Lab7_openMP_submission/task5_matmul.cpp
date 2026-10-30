#include <omp.h>
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

// Matrices are flattened 1D vectors: element (i, j) is at i * n + j

void matmul_serial(const vector<double>& A, const vector<double>& B, vector<double>& C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

void matmul_outer(const vector<double>& A, const vector<double>& B, vector<double>& C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

void matmul_collapse(const vector<double>& A, const vector<double>& B, vector<double>& C, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

// Bonus: loop order i, k, j
void matmul_serial_ikj(const vector<double>& A, const vector<double>& B, vector<double>& C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) C[i * n + j] = 0.0;
        for (int k = 0; k < n; k++) {
            double a_ik = A[i * n + k];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a_ik * B[k * n + j];
        }
    }
}

void matmul_outer_ikj(const vector<double>& A, const vector<double>& B, vector<double>& C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) C[i * n + j] = 0.0;
        for (int k = 0; k < n; k++) {
            double a_ik = A[i * n + k];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a_ik * B[k * n + j];
        }
    }
}

bool same_matrix(const vector<double>& X, const vector<double>& Y) {
    for (size_t k = 0; k < X.size(); k++)
        if (fabs(X[k] - Y[k]) > 1e-9 * (1.0 + fabs(X[k]))) return false;
    return true;
}

void print_row(const char* name, int threads, double time, double base, bool ok) {
    double speedup = base / time;
    printf("%-14s %7d %10.4f %9.2f %11.2f   %s\n", name, threads, time,
           speedup, speedup / threads, ok ? "YES" : "NO");
}

int main() {
    int sizes[] = {256, 512, 1024};
    int thread_counts[] = {1, 2, 4, 8};

    printf("Matrices are flattened 1D vectors. Threads set with omp_set_num_threads().\n\n");

    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        size_t total = (size_t)n * n;
        vector<double> A(total), B(total), C_ref(total), C(total), C_ikj_ref(total);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = (i + j) % 10;
                B[i * n + j] = (i * 2 + j) % 7;
            }
        }

        double t0 = omp_get_wtime();
        matmul_serial(A, B, C_ref, n);
        double serial_ijk = omp_get_wtime() - t0;

        t0 = omp_get_wtime();
        matmul_serial_ikj(A, B, C_ikj_ref, n);
        double serial_ikj = omp_get_wtime() - t0;

        printf("N = %d\n", n);
        printf("Serial i-j-k: %.4f s    Serial i-k-j (bonus): %.4f s    (ikj result correct: %s)\n",
               serial_ijk, serial_ikj, same_matrix(C_ref, C_ikj_ref) ? "YES" : "NO");
        printf("Speedup/efficiency of ijk versions are relative to serial i-j-k,\n");
        printf("speedup/efficiency of the ikj version is relative to serial i-k-j.\n");
        printf("Version        Threads    Time(s)   Speedup  Efficiency   Correct\n");

        for (int t = 0; t < 4; t++) {
            int threads = thread_counts[t];
            omp_set_num_threads(threads);

            for (size_t k = 0; k < total; k++) C[k] = -1;
            double start = omp_get_wtime();
            matmul_outer(A, B, C, n);
            double time = omp_get_wtime() - start;
            print_row("outer-i", threads, time, serial_ijk, same_matrix(C_ref, C));

            for (size_t k = 0; k < total; k++) C[k] = -1;
            start = omp_get_wtime();
            matmul_collapse(A, B, C, n);
            time = omp_get_wtime() - start;
            print_row("collapse(2)", threads, time, serial_ijk, same_matrix(C_ref, C));

            for (size_t k = 0; k < total; k++) C[k] = -1;
            start = omp_get_wtime();
            matmul_outer_ikj(A, B, C, n);
            time = omp_get_wtime() - start;
            print_row("outer-i (ikj)", threads, time, serial_ikj, same_matrix(C_ref, C));
        }
        printf("\n");
    }
    return 0;
}
