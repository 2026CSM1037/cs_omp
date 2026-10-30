#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

// Every element x is updated "work[i]" times: more work = slower element.

void run_serial(const vector<double>& in, const vector<int>& work, vector<double>& out, int n) {
    for (int i = 0; i < n; i++) {
        double x = in[i];
        for (int j = 0; j < work[i]; j++)
            x = x * 1.0000001 + 0.0000001;
        out[i] = x;
    }
}

void run_static(const vector<double>& in, const vector<int>& work, vector<double>& out, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double x = in[i];
        for (int j = 0; j < work[i]; j++)
            x = x * 1.0000001 + 0.0000001;
        out[i] = x;
    }
}

void run_dynamic(const vector<double>& in, const vector<int>& work, vector<double>& out, int n, int chunk) {
    #pragma omp parallel for schedule(dynamic, chunk)
    for (int i = 0; i < n; i++) {
        double x = in[i];
        for (int j = 0; j < work[i]; j++)
            x = x * 1.0000001 + 0.0000001;
        out[i] = x;
    }
}

void run_guided(const vector<double>& in, const vector<int>& work, vector<double>& out, int n) {
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; i++) {
        double x = in[i];
        for (int j = 0; j < work[i]; j++)
            x = x * 1.0000001 + 0.0000001;
        out[i] = x;
    }
}

// type 0 = uniform, 1 = uneven (sawtooth, like the sample), 2 = uneven (ramp: heavy at the end)
// "avg" is the average number of loop repetitions per element
void make_workload(vector<int>& work, int n, int type, long long avg) {
    for (int i = 0; i < n; i++) {
        if (type == 0)      work[i] = (int)avg;
        else if (type == 1) work[i] = (int)((i % 1000) * (2 * avg) / 1000);
        else                work[i] = (int)((long long)i * (2 * avg) / n);
    }
}

void print_cell(double time, double serial_time, int threads) {
    double speedup = serial_time / time;
    printf("  %7.4f / %5.2f / %4.2f ", time, speedup, speedup / threads);
}

int main(int argc, char* argv[]) {
    long long total_work = 250000000;
    if (argc > 1) total_work = atoll(argv[1]);

    int sizes[] = {100000, 1000000, 10000000};
    int thread_counts[] = {1, 2, 4, 8};
    const char* type_names[] = {"UNIFORM workload", "UNEVEN workload (sawtooth: i % 1000)",
                                "UNEVEN workload (ramp: heavier toward the end)"};
    const int CHUNK = 100;

    printf("Threads are set with omp_set_num_threads() in the code.\n");
    printf("Total work is about %lld loop repetitions for every N (change it with an argument).\n", total_work);
    printf("Dynamic chunk size = %d. Each cell is: time (s) / speedup / efficiency\n\n", CHUNK);

    // Part 1: static vs dynamic vs guided
    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        long long avg = total_work / n;
        vector<double> in(n), out_serial(n), out(n);
        vector<int> work(n);
        for (int i = 0; i < n; i++) in[i] = i * 0.001;

        for (int type = 0; type < 3; type++) {
            make_workload(work, n, type, avg);

            double start = omp_get_wtime();
            run_serial(in, work, out_serial, n);
            double serial_time = omp_get_wtime() - start;

            printf("N = %d, %s   (serial = %.4f s)\n", n, type_names[type], serial_time);
            printf("Threads        Static                  Dynamic                 Guided          Correct\n");

            for (int t = 0; t < 4; t++) {
                int threads = thread_counts[t];
                omp_set_num_threads(threads);
                bool ok = true;
                double times[3];

                start = omp_get_wtime();
                run_static(in, work, out, n);
                times[0] = omp_get_wtime() - start;
                if (out != out_serial) ok = false;

                start = omp_get_wtime();
                run_dynamic(in, work, out, n, CHUNK);
                times[1] = omp_get_wtime() - start;
                if (out != out_serial) ok = false;

                start = omp_get_wtime();
                run_guided(in, work, out, n);
                times[2] = omp_get_wtime() - start;
                if (out != out_serial) ok = false;

                printf("%7d", threads);
                for (int v = 0; v < 3; v++) print_cell(times[v], serial_time, threads);
                printf("  %s\n", ok ? "YES" : "NO");
            }
            printf("\n");
        }
    }

    // Part 2: chunk size of the dynamic schedule
    {
        int n = 1000000;
        long long avg = total_work / n;
        vector<double> in(n), out_serial(n), out(n);
        vector<int> work(n);
        for (int i = 0; i < n; i++) in[i] = i * 0.001;
        int chunks[] = {1, 10, 100, 1000, 10000, 100000};

        for (int type = 0; type < 3; type += 2) {
            make_workload(work, n, type, avg);
            double start = omp_get_wtime();
            run_serial(in, work, out_serial, n);
            double serial_time = omp_get_wtime() - start;

            printf("PART 2: chunk size of schedule(dynamic, chunk), N = %d, %s\n", n, type_names[type]);
            printf("Serial = %.4f s. Each cell is: time (s) / speedup\n", serial_time);
            printf("  Chunk       4 threads        8 threads     Correct\n");

            for (int c = 0; c < 6; c++) {
                printf("%7d", chunks[c]);
                bool ok = true;
                int counts[2] = {4, 8};
                for (int t = 0; t < 2; t++) {
                    omp_set_num_threads(counts[t]);
                    start = omp_get_wtime();
                    run_dynamic(in, work, out, n, chunks[c]);
                    double time = omp_get_wtime() - start;
                    if (out != out_serial) ok = false;
                    printf("   %7.4f / %5.2f", time, serial_time / time);
                }
                printf("     %s\n", ok ? "YES" : "NO");
            }
            printf("\n");
        }
    }
    return 0;
}
