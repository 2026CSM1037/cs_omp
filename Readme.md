# CS527 – OpenMP Labs 8 and 9 (General README)

This README covers all the programs of **Lab 8** (arrays and matrices) and **Lab 9** (tasks, scheduling, reductions). It explains for each program what it is trying to achieve, what goes in, what you should see come out, and how to read the results.

- [1. Quick overview](#1-quick-overview)
- [2. Requirements and how to compile and run](#2-requirements-and-how-to-compile-and-run)
- [3. Input: what the programs need](#3-input-what-the-programs-need)
- [4. How to read the output](#4-how-to-read-the-output)
- [5. Lab 8 programs](#5-lab-8-programs)
- [6. Lab 9 programs](#6-lab-9-programs)
- [7. OpenMP cheat sheet](#7-openmp-cheat-sheet)
- [8. Measurement caveats](#8-measurement-caveats)
- [9. Troubleshooting](#9-troubleshooting)

---

## 1. Quick overview

OpenMP lets one program use several CPU cores. You add a line starting with `#pragma omp ...` above a loop or a function call, and the compiler splits the work among **threads**. Every program here:

1. builds its own test data (nothing to type in),
2. runs a **serial** version (one thread, the baseline),
3. runs one or more **OpenMP** versions with 1, 2, 4 and 8 threads,
4. checks that the parallel answer equals the serial answer,
5. prints timing tables with **speedup** and (where asked) **efficiency**.

| Lab | File | What it achieves |
|---|---|---|
| 8 | `task1_search.cpp` | Finds whether a key exists in an array and its first index, in parallel |
| 8 | `task2_sort.cpp` | Parallel bubble sort (odd-even transposition sort) and its synchronization |
| 8 | `task3_histogram.cpp` | Counts values 0–9 without race conditions (5 ways) |
| 8 | `task4_transpose.cpp` | Transposes a matrix (outer loop, `collapse(2)`, blocked) |
| 8 | `task5_convolution.cpp` | Applies a 3x3 filter to an image (2D convolution) |
| 9 | `task1_mergesort.cpp` | Recursive parallel merge sort with `omp task` / `taskwait` and a cutoff |
| 9 | `task2_schedule.cpp` | Compares `static`, `dynamic`, `guided` scheduling on uneven work |
| 9 | `task3_evenodd.cpp` | Counts even/odd numbers: two reductions in one loop vs `atomic` |

Suggested folder layout for git / submission:

```
cs527_openmp/
├── README.md              (this file)
├── lab8/
│   ├── task1_search.cpp  task2_sort.cpp  task3_histogram.cpp
│   └── task4_transpose.cpp  task5_convolution.cpp
└── lab9/
    └── task1_mergesort.cpp  task2_schedule.cpp  task3_evenodd.cpp
```

---

## 2. Requirements and how to compile and run

**You need:** a C++ compiler with OpenMP support (`g++`), nothing else.

### Ubuntu (lab machines)

`g++` is normally installed already (otherwise `sudo apt install g++`).

### Windows

Use the **MSYS2 "UCRT64" terminal**, a 64-bit `g++`. Check it with `g++ --version`, which should mention `x86_64`.
- Do **not** use the old `c:\mingw` (32-bit) compiler. It fails with `cannot find -lpthread` and prints `%lld` wrongly.
- If you get `cannot find -lgomp`, install the OpenMP library: `pacman -S mingw-w64-ucrt-x86_64-libgomp` (or `pacman -S mingw-w64-ucrt-x86_64-toolchain`).
- A folder name with spaces needs quotes: `cd "/c/Users/you/Documents/My Folder"`.
- An alternative is WSL (Ubuntu inside Windows), which behaves exactly like the lab machines.

### Compile one program

The same command works on Windows and Ubuntu:

```bash
g++ -fopenmp -O2 -Wall task1_search.cpp -o task1
./task1                # on Windows also: ./task1.exe
```

It should compile with **no warnings**. The flags: `-fopenmp` turns OpenMP on (without it the `#pragma omp` lines are ignored and everything runs serially), `-O2` turns on normal optimizations, `-Wall` shows warnings.

### Compile and run all programs of one lab

Compile first, run after the loop (running inside the loop runs everything several times):

```bash
# Lab 8
for t in task1_search task2_sort task3_histogram task4_transpose task5_convolution; do
  g++ -fopenmp -O2 -Wall $t.cpp -o $t
done
./task1_search; ./task2_sort; ./task3_histogram; ./task4_transpose; ./task5_convolution

# Lab 9
for t in task1_mergesort task2_schedule task3_evenodd; do
  g++ -fopenmp -O2 -Wall $t.cpp -o $t
done
./task1_mergesort; ./task2_schedule; ./task3_evenodd
```

To keep a full copy of the output for your report, save it to a file:

```bash
./task2_schedule > task2_output.txt
```

(This also protects you against losing lines of a long output in the terminal scroll-back.)

**Number of threads.** It is set inside every program with `omp_set_num_threads()`. Each program loops over **1, 2, 4 and 8 threads** by itself, so you do not need `OMP_NUM_THREADS`. Your machine may have fewer than 8 cores; the programs still run, but the speedup stops growing at the number of cores.

---

## 3. Input: what the programs need

**There is no input file and nothing to type.** Each program creates its own data inside the code, with a fixed random seed (`mt19937` seed `12345`), so the data are the same on every run and the serial and parallel versions can be compared.

You may give **one optional number** on the command line to limit memory or run time:

| Program | Optional argument | Default | Example |
|---|---|---|---|
| Lab 8 `task1_search` | biggest array size N | 100000000 | `./task1_search 10000000` |
| Lab 8 `task2_sort` | biggest array size N | 40000 | `./task2_sort 20000` |
| Lab 8 `task3_histogram` | biggest array size N | 100000000 | `./task3_histogram 10000000` |
| Lab 8 `task4_transpose` | biggest matrix side N | 8192 | `./task4_transpose 4096` |
| Lab 8 `task5_convolution` | biggest image side N | 8192 | `./task5_convolution 4096` |
| Lab 9 `task1_mergesort` | biggest array size N (the cutoff experiment needs at least 1000000) | 10000000 | `./task1_mergesort 1000000` |
| Lab 9 `task2_schedule` | total amount of work (loop repetitions) per N | 250000000 | `./task2_schedule 50000000` |
| Lab 9 `task3_evenodd` | biggest array size N | 100000000 | `./task3_evenodd 10000000` |

Sizes bigger than the argument are skipped. If the machine runs out of memory, the program prints `Not enough memory ... skipping` and goes on.

**Memory and time to expect with the defaults**

- N = 10^8 ints (Lab 8 Task 1 and 3, Lab 9 Task 3) needs about 400 MB for the array.
- 8192 x 8192 doubles (Lab 8 Task 4 and 5) need about 1.5 GB in total.
- Slow on purpose: the `critical` / `atomic` versions at N = 10^8 (seconds each), and Lab 8 Task 2 at N = 40000 (a few seconds per run). Lab 9 Task 2 takes a couple of minutes with the default work.

---

## 4. How to read the output

| Word in the output | Meaning |
|---|---|
| **Serial** | The baseline: the same job with one thread and no OpenMP |
| **Parallel / Time** | Time of the OpenMP version. Seconds `(s)` or milliseconds `(ms)`, as written in the heading |
| **Speedup** | `serial time / parallel time`. 4.0 means 4 times faster; below 1.0 means slower than serial |
| **Efficiency** | `speedup / number of threads`. 1.0 is perfect, lower means threads are wasting time |
| **Correct / Match / Sorted correctly** | `YES` when the parallel result equals the serial (or `std::sort`) result. **A `NO` here means a bug or a race condition** (except for the deliberately broken versions, see Section 5) |
| **N** | Problem size (array length or matrix/image side) |

General expectations:

- The `Correct` column must be **YES** on every row.
- With 1 thread, the parallel version should be about as fast as serial (speedup near 1.0).
- Speedup usually **increases with threads and with N**, but not perfectly. Many of these jobs are limited by memory speed, and small jobs are dominated by thread start-up cost.

---

## 5. Lab 8 programs

All Lab 8 matrices are stored as **flattened 1D vectors**: element (i, j) is at `A[i * n + j]`.

### 5.1 Task 1 – Array search (`task1_search.cpp`)

**Goal.** Find out whether a key exists in an array and, if so, the index of its first occurrence.

**Input.** Created internally: N random integers (N = 10^7 and 10^8). The key (`2000000000`, which cannot occur by chance) is planted at chosen positions: 10% of the array, the middle, the very end, twice (50% and 75%), or not at all.

**Method.** Serial: a loop that stops with `break`. Parallel: `break` is not allowed in `parallel for`, so every element is checked and `reduction(min:first)` keeps the smallest matching index, which is the first occurrence.

**Expected output** (excerpt from a real run):

```
N = 10000000
Case                   Threads  Serial(ms)  Parallel(ms)  Speedup        Index  Match
key at 10% of array          1       0.200         3.100     0.06      1000000  YES
key in the middle            8       2.000         0.850     2.35      5000000  YES
key at the very end          8       4.000         0.950     4.21      9999999  YES
key twice (50%, 75%)         8       2.000         0.900     2.22      5000000  YES
key not present              8       4.050         0.900     4.50           -1  YES
```

**What to check.**
- `Index` is N/10, N/2, N-1, N/2 (the *first* of the two), and `-1` when the key is not present.
- When the key is early, serial can win (it stops early, parallel scans everything). When the whole array has to be scanned, parallel wins (about 4 to 5 times on 8 threads).

### 5.2 Task 2 – Sorting (`task2_sort.cpp`)

**Goal.** Parallelize bubble sort and explain the synchronization required.

**Input.** N random integers (N = 10000, 20000, 40000).

**Method.** Bubble sort cannot be parallelized directly. The parallel version is **odd-even transposition sort**: in even phases the pairs (0,1), (2,3), ... are compared, in odd phases the pairs (1,2), (3,4), ... Inside one phase all pairs are independent (no race, no lock needed). Between phases a **barrier** (the implicit one at the end of `#pragma omp for`) makes sure phase p is finished before phase p+1 starts.

**Expected output** (excerpt):

```
N = 20000
Serial bubble sort:        0.1680 s (correct: YES)
Serial odd-even sort:      0.0930 s (correct: YES)
std::sort (for reference): 0.0010 s
Threads   Parallel(s)   Speedup vs bubble   Speedup vs serial odd-even   Efficiency   Correct
      1        0.0920                1.83                         1.01         1.01   YES
      8        0.7340                0.23                         0.13         0.02   YES
```

**What to check.** Every row sorted correctly (compared with `std::sort`). Use "speedup vs serial odd-even" to judge OpenMP itself, since odd-even is a different (more CPU-friendly) algorithm than bubble sort. Do not be surprised if more threads make it **slower**: there is a barrier after each of the N phases, and the work per phase is small.

### 5.3 Task 3 – Histogram (`task3_histogram.cpp`)

**Goal.** Count how often each value 0 to 9 appears in a large array without race conditions.

**Input.** N random integers in 0..9 (N = 10^7 and 10^8).

**Methods compared.**

| Version | Idea | Expected |
|---|---|---|
| Naive | `hist[a[i]]++`, no protection | **Wrong** (only shown in Part 1) |
| `atomic` | each increment is indivisible | Correct, slow |
| `critical` | one thread at a time | Correct, slowest |
| Local copy | each thread has private counters, merged once at the end | Correct, fast |
| `reduction(+:hist[0:10])` | same idea, done by OpenMP | Correct, fast |

**Expected output.** Part 1 shows the race (naive totals are far below N), Part 2 shows times and speedups of the correct versions:

```
PART 1: race condition demo (N = 10000000, 4 threads)
Value                  0         1  ...         9
Serial           1000266   1002109  ...   1000880   total = 10000000
Naive run 1       266217    265963  ...    264984   total = 2646027

PART 2: N = 100000000   (serial time = 0.048000 s)
Threads    Atomic   Critical  LocalCopy  Reduction   Correct
      1    0.4240     1.1970     0.0480     0.0490   YES
      8    1.7180     4.7850     0.0080     0.0070   YES
```

**What to check.** The serial total is exactly N. The naive totals are **less than N** (lost updates) and differ from run to run; this is the intended demonstration. All four correct versions give `YES`. `atomic` and `critical` are slower than serial; local copy and `reduction` scale (about 6 to 7 times on 8 threads).

### 5.4 Task 4 – Matrix transpose (`task4_transpose.cpp`)

**Goal.** Compute `B[j][i] = A[i][j]` in parallel.

**Input.** An N x N matrix of doubles (N = 1024, 4096, 8192) filled with `A[k] = k`.

**Versions.** Parallel on the outer loop; `collapse(2)` (both loops merged); **blocked** (bonus: works on 32 x 32 tiles so the data stays in the cache).

**Expected output** (excerpt):

```
N = 4096   (serial = 210.000 ms)
Threads | Outer(ms) Collapse(ms) Blocked(ms) | Spd(outer) Spd(collapse) Spd(blocked) | Correct
      1 |   135.500      147.000     163.000 |       1.55          1.43         1.29 | YES
      8 |    30.000       29.000      20.500 |       7.00          7.24        10.24 | YES
```

**What to check.** `Correct = YES` (each version equals the serial transpose). Times are averages over repetitions. Expect the blocked version to be best on big matrices. A speedup above the thread count is possible because blocking also improves cache use compared to the serial code.

### 5.5 Task 5 – 2D convolution (`task5_convolution.cpp`)

**Goal.** Slide a 3x3 filter over an image and compute a new value for every position where the filter fits completely (no padding). An N x N image gives an **(N-2) x (N-2)** output.

**How one output pixel is computed.** Multiply the 9 pixels under the filter by the 9 filter numbers and add them. With the averaging filter (every weight 1/9) this is the mean of the 9 pixels, which blurs the image.

**Input.** Part 1: the 5x5 image of the assignment (values 10, 20, ..., 250). Part 2: an N x N image (N = 1024, 4096, 8192) filled with a repeating pattern, and the averaging filter.

**Expected output.** Part 1 must reproduce the assignment example exactly:

```
Averaging filter (blur), output:
   70.00   80.00   90.00
  120.00  130.00  140.00
  170.00  180.00  190.00
Parallel result matches serial: YES
```

The first value is 630/9 = 70. The sharpen filter leaves this smooth image unchanged, and the edge-detection filter gives all zeros (a straight gradient has no edges). Part 2:

```
Image 4096 x 4096   (serial = 28.500 ms)
Threads   Outer(ms)   Collapse(ms)   Speedup(outer)   Speedup(collapse)   Correct
      1      32.000         36.500             0.89                0.78   YES
      8       6.500          7.500             4.38                3.80   YES
```

**What to check.** The 5x5 output above, all `YES`, and speedup improving with image size. There is no race, because each thread only reads the image and writes its own output pixels.

---

## 6. Lab 9 programs

### 6.1 Task 1 – Merge sort with tasks (`task1_mergesort.cpp`)

**Goal.** Learn recursive task parallelism with `#pragma omp task` and `#pragma omp taskwait`.

**Input.** N random integers (N = 10^5, 10^6, 10^7).

**Method.** Merge sort splits the array in two halves, sorts each half and merges them. The two halves are independent, so each becomes a task. The parent waits with `taskwait` for both children before it merges. Pieces smaller than the **cutoff** (2048) are sorted serially, so tiny tasks are not created. A temporary array is shared; every call works on its own range `left..right`. The recursion starts inside `#pragma omp parallel` + `#pragma omp single`.

**Expected output.** Part 1 (times, speedup, efficiency) and Part 2 (effect of the cutoff, N = 10^6):

```
N = 10000000   (serial = 842.000 ms, sorted correctly: YES)
Threads   Parallel(ms)   Speedup   Efficiency   Sorted+Matches serial
      1       842.000      1.00         1.00   YES
      8       199.000      4.23         0.53   YES

PART 2: effect of the cutoff (N = 1000000, serial = 74.000 ms)
Each cell is: parallel time in ms (speedup)
Cutoff    1 thread          2 threads         4 threads         8 threads         Correct
     4     108.667 ( 0.68)     141.000 ( 0.52)     200.667 ( 0.37)     281.667 ( 0.26)   YES
  2048      74.000 ( 1.00)      41.000 ( 1.80)      29.333 ( 2.52)      20.667 ( 3.58)   YES
500000      74.000 ( 1.00)      40.000 ( 1.85)      41.333 ( 1.79)      39.000 ( 1.90)   YES
```

**What to check.** Always `YES` (sorted and equal to serial). Speedup of about 3 to 4 on 8 threads. A tiny cutoff (like 4) makes the sort slower than serial (too many tiny tasks), a huge cutoff (like 500000) creates only two tasks, and a wide middle range of cutoffs works well.

### 6.2 Task 2 – Scheduling (`task2_schedule.cpp`)

**Goal.** See how `static`, `dynamic` and `guided` scheduling affect load balancing.

**Input.** An array of N doubles (N = 10^5, 10^6, 10^7). Every element is updated `work[i]` times, so a larger `work[i]` means a slower iteration. Three workloads: **uniform**, **uneven sawtooth** (`i % 1000`, as in the sample) and **uneven ramp** (the last elements are the slowest). The total work is about 250 million repetitions for every N, so the sample's much bigger work (minutes to hours at N = 10^7) is avoided; change it with the optional argument.

**The three schedules.**

| Schedule | How the iterations are handed out |
|---|---|
| `static` | Equal blocks decided before the loop starts. No overhead, but no balancing |
| `dynamic, chunk` | Chunks of `chunk` iterations given on request. Balances uneven work, but each request costs time |
| `guided` | Like dynamic, with big chunks first and smaller ones later, so fewer requests |

**Expected output.** Part 1 tables (each cell is `time (s) / speedup / efficiency`) and Part 2 (chunk size):

```
N = 10000000, UNEVEN workload (ramp: heavier toward the end)   (serial = 0.0860 s)
Threads        Static                  Dynamic                 Guided          Correct
      8   0.0290 /  2.97 / 0.37    0.0160 /  5.37 / 0.67    0.0120 /  7.17 / 0.90   YES

PART 2: chunk size of schedule(dynamic, chunk), N = 1000000, UNIFORM workload
Serial = 0.2560 s. Each cell is: time (s) / speedup
  Chunk       4 threads        8 threads     Correct
      1    0.1050 /  2.44    0.0550 /  4.65     YES
    100    0.0660 /  3.88    0.0330 /  7.76     YES
 100000    0.0780 /  3.28    0.0520 /  4.92     YES
```

**What to check.** All `YES`. On the ramp workload `static` should be clearly the worst (the thread with the last block gets the most work). On the uniform and sawtooth workloads the schedules are close. Chunk size 1 is slow (scheduling overhead), a huge chunk is slow (poor balance), medium chunks are best.

### 6.3 Task 3 – Even / odd count (`task3_evenodd.cpp`)

**Goal.** Practise several reductions in one loop and compare `reduction` with `atomic`.

**Input.** N random integers (N = 10^5, 10^6, 10^7, 10^8).

**Methods.**
- `reduction(+:even, odd)`: each thread has private counters, added together at the end.
- `#pragma omp atomic` on shared `even` and `odd`: correct, but every update has to wait for the others.

**Expected output** (excerpt):

```
N = 100000000   (serial = 285.000 ms, even = 49993431, odd = 50006569, even + odd = N: YES)
Threads | Reduction(ms) Speedup Efficiency | Atomic(ms) Speedup Efficiency | Correct
      1 |       293.000    0.97       0.97 |    574.000    0.50       0.50 | YES
      8 |        66.000    4.32       0.54 |   1908.000    0.15       0.02 | YES
```

**What to check.** `even + odd = N` is `YES` for every size, and `Correct = YES`. The reduction version gets faster with more threads; the atomic version is slower than serial and gets slower as threads are added.

---

## 7. OpenMP cheat sheet

| Directive | What it does | Used in |
|---|---|---|
| `#pragma omp parallel for` | Splits the loop iterations among threads (iterations must be independent) | most tasks |
| `reduction(+:x)` / `min` / `max` | Each thread gets a private copy of `x`, combined at the end. Several variables: `reduction(+:a, b)` | search, histogram, even/odd |
| `atomic` | One single update of a shared variable is indivisible | histogram, even/odd |
| `critical` | Only one thread at a time runs the block | histogram |
| `collapse(2)` | Merges two nested loops into one big loop to share | transpose, convolution |
| `schedule(static / dynamic, c / guided)` | How loop iterations are handed out | Lab 9 Task 2 |
| `#pragma omp parallel` + `#pragma omp for` | Create threads once, share several loops inside | odd-even sort |
| `omp task` / `taskwait` | Create a piece of work for any thread / wait for the child tasks | merge sort |
| `omp_set_num_threads(n)` | Sets the number of threads | all programs |
| `omp_get_wtime()` | Wall-clock time in seconds | all programs |

**Data race:** two threads read and write the same variable at the same time, so updates are lost. Fixes: private copies and `reduction` (best), `atomic`, or `critical`.

---

## 8. Measurement caveats

- **Timer resolution.** On Windows `omp_get_wtime()` ticks about once per millisecond. Times of a few milliseconds are coarse, and `nan` / `inf` / 0.00 speedups can appear for tiny jobs. The programs repeat short jobs and print the average where possible.
- **Run-to-run variation.** Single runs can vary by 10% or more. Run each program two or three times and report a typical run, and take final numbers on one machine (your laptop or the lab PC) and say which.
- **Serial baseline.** If a serial time is much smaller than the 1-thread parallel time for the same job (for example 10 times faster), the compiler may have optimised the repeated serial calls, so compare with the 1-thread parallel time instead and mention it in the report.
- **Hyperthreads and core types.** Your CPU may report more threads than physical cores (or have fast and slow cores), so speedup flattens above the number of real cores, and `static` scheduling can look worse than expected.
- **Memory bound jobs.** Search, transpose, convolution, vector-style loops do very little arithmetic per byte read, so more threads help only up to the speed of the memory.
- **Race demos are intentional.** The naive histogram (Lab 8 Task 3) is *supposed* to give wrong totals.

---

## 9. Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `cannot find -lpthread` and `unknown conversion type character 'l'` warnings | You are using the old 32-bit `c:\mingw` compiler. Use the MSYS2 UCRT64 terminal (Section 2) |
| `cannot find -lgomp` | The OpenMP library is missing: `pacman -S mingw-w64-ucrt-x86_64-libgomp` |
| `No such file or directory` when running a program | It was not compiled (or compiled under another name). Compile first, then run |
| Everything runs on one thread, speedup is always about 1 | `-fopenmp` was left out of the compile command |
| `Not enough memory ... skipping` | Give a smaller maximum size, e.g. `./task1_search 10000000` |
| The program seems stuck | The `critical` / `atomic` versions at N = 10^8, or Lab 9 Task 2, are slow; wait, or use the optional argument |
| A `Correct` column shows `NO` | The parallel result differs from serial. Check the loop for a shared variable written by several threads |
| Output scrolled away in the terminal | Save it with `./taskX > output.txt` |

---

## What to submit

For each task: the `.cpp` file, the timing tables copied from the program output, and a 2 to 3 line explanation of the speedup (or lack of it), plus the answers to the questions in the assignment. State that the number of threads is set with `omp_set_num_threads()` in the code, and the CPU and number of cores of your machine. Compile everything with `-fopenmp -Wall` and check that there are no warnings.
