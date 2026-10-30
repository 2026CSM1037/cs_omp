#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <new>
using namespace std;

// Image is n x n, flattened 1D: pixel (i, j) is img[i * n + j].
// No padding, so the output is (n-2) x (n-2): out[i * (n-2) + j].
// The 3x3 filter is stored as 9 numbers: k[row * 3 + col].

void conv_serial(const vector<double>& img, vector<double>& out, int n, const vector<double>& k) {
    int m = n - 2;
    double w[9];
    for (int p = 0; p < 9; p++) w[p] = k[p];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++)
                    sum += img[(i + ki) * n + (j + kj)] * w[ki * 3 + kj];
            out[i * m + j] = sum;
        }
    }
}

void conv_outer(const vector<double>& img, vector<double>& out, int n, const vector<double>& k) {
    int m = n - 2;
    #pragma omp parallel
    {
        double w[9];   // every thread keeps its own copy of the filter
        for (int p = 0; p < 9; p++) w[p] = k[p];

        #pragma omp for
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0.0;
                for (int ki = 0; ki < 3; ki++)
                    for (int kj = 0; kj < 3; kj++)
                        sum += img[(i + ki) * n + (j + kj)] * w[ki * 3 + kj];
                out[i * m + j] = sum;
            }
        }
    }
}

void conv_collapse(const vector<double>& img, vector<double>& out, int n, const vector<double>& k) {
    int m = n - 2;
    #pragma omp parallel
    {
        double w[9];
        for (int p = 0; p < 9; p++) w[p] = k[p];

        #pragma omp for collapse(2)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0.0;
                for (int ki = 0; ki < 3; ki++)
                    for (int kj = 0; kj < 3; kj++)
                        sum += img[(i + ki) * n + (j + kj)] * w[ki * 3 + kj];
                out[i * m + j] = sum;
            }
        }
    }
}

bool same_image(const vector<double>& x, const vector<double>& y) {
    for (size_t p = 0; p < x.size(); p++)
        if (fabs(x[p] - y[p]) > 1e-9 * (1.0 + fabs(x[p]))) return false;
    return true;
}

void print_matrix(const vector<double>& m, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%8.2f", m[i * cols + j]);
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    int max_n = 8192;
    if (argc > 1) max_n = atoi(argv[1]);

    vector<double> blur(9, 1.0 / 9.0);
    vector<double> sharpen = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    vector<double> edge = {-1, -1, -1, -1, 8, -1, -1, -1, -1};

    // Part 1: the 5x5 example from the assignment
    {
        int n = 5;
        vector<double> img(n * n);
        for (int p = 0; p < n * n; p++) img[p] = 10.0 * (p + 1);

        printf("PART 1: 5x5 example (3x3 output, no padding)\n");
        printf("Input image:\n");
        print_matrix(img, n, n);

        const char* names[3] = {"Averaging filter (blur)", "Sharpen filter", "Edge-detection filter"};
        const vector<double>* filters[3] = {&blur, &sharpen, &edge};
        for (int f = 0; f < 3; f++) {
            vector<double> out_s(9), out_p(9);
            conv_serial(img, out_s, n, *filters[f]);
            omp_set_num_threads(4);
            conv_collapse(img, out_p, n, *filters[f]);
            printf("\n%s, output:\n", names[f]);
            print_matrix(out_s, 3, 3);
            printf("Parallel result matches serial: %s\n", same_image(out_s, out_p) ? "YES" : "NO");
        }
        printf("\n");
    }

    // Part 2: timing on big images (averaging filter)
    int sizes[] = {1024, 4096, 8192};
    int thread_counts[] = {1, 2, 4, 8};

    printf("PART 2: timing with the averaging filter. Threads set with omp_set_num_threads().\n");
    printf("Times are averages over several repetitions, in milliseconds.\n\n");

    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        if (n > max_n) continue;
        int m = n - 2;
        int reps = (int)(50000000LL / ((long long)n * n));
        if (reps < 1) reps = 1;

        try {
            vector<double> img((size_t)n * n);
            vector<double> ref((size_t)m * m), out((size_t)m * m);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    img[(size_t)i * n + j] = (i * 7 + j * 13) % 256;

            conv_serial(img, ref, n, blur);
            double start = omp_get_wtime();
            for (int r = 0; r < reps; r++) conv_serial(img, ref, n, blur);
            double serial_ms = (omp_get_wtime() - start) / reps * 1000.0;

            printf("Image %d x %d   (serial = %.3f ms)\n", n, n, serial_ms);
            printf("Threads   Outer(ms)   Collapse(ms)   Speedup(outer)   Speedup(collapse)   Correct\n");

            for (int t = 0; t < 4; t++) {
                omp_set_num_threads(thread_counts[t]);

                fill(out.begin(), out.end(), -1.0);
                conv_outer(img, out, n, blur);
                bool ok = same_image(ref, out);
                fill(out.begin(), out.end(), -1.0);
                conv_collapse(img, out, n, blur);
                if (!same_image(ref, out)) ok = false;

                start = omp_get_wtime();
                for (int r = 0; r < reps; r++) conv_outer(img, out, n, blur);
                double outer_ms = (omp_get_wtime() - start) / reps * 1000.0;

                start = omp_get_wtime();
                for (int r = 0; r < reps; r++) conv_collapse(img, out, n, blur);
                double collapse_ms = (omp_get_wtime() - start) / reps * 1000.0;

                printf("%7d   %9.3f   %12.3f   %14.2f   %17.2f   %s\n", thread_counts[t],
                       outer_ms, collapse_ms, serial_ms / outer_ms, serial_ms / collapse_ms,
                       ok ? "YES" : "NO");
            }
            printf("\n");
        } catch (bad_alloc&) {
            printf("Not enough memory for %d x %d, skipping.\n\n", n, n);
        }
    }
    return 0;
}
