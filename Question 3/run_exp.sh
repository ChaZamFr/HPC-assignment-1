#!/usr/bin/env bash
# run_exp.sh
# Usage: ./run_exp.sh ./hardy_ram <n_hr> <maxA> <thread1> [thread2 ...]

if [ $# -lt 4 ]; then
    echo "Usage: $0 <program> <n_hr> <maxA> <thread1> [thread2 ...]"
    exit 1
fi

PROGRAM=$1
N_HR=$2
MAXA=$3
shift 3
THREADS=("$@")
REPEATS=7

OUT_COMBINED="timing_runs.csv"
NUMBERS_DIR="HR numbers"
RUNS_DIR="runs"

if [ -d "$NUMBERS_DIR"]; then
    rm -r "$NUMBERS_DIR"
fi

mkdir -p "$NUMBERS_DIR"

if [ -d "$RUNS_DIR"]; then
    rm -r "$RUNS_DIR"
fi

mkdir -p "$RUNS_DIR"

echo "Run,Threads,Sequential,Static,Dynamic,Guided,Task,Speedup_Static,Speedup_Dynamic,Speedup_Guided,Speedup_Task" > "$OUT_COMBINED"

for t in "${THREADS[@]}"; do
    echo ">>> Running thread count: $t"
    for run in $(seq 1 $REPEATS); do
        echo "---- Run $run / $REPEATS ----"
        # Run the program with one thread count
        $PROGRAM "$N_HR" "$MAXA" "$t" > prog_run.log 2>&1

        # Ensure hr_times.csv exists
        if [ ! -f hr_times.csv ]; then
            echo "Error: hr_times.csv not produced. See prog_run.log"
            exit 2
        fi

        # Append each line of hr_times.csv to timing_runs.csv with Run number
        # Skip header line
        tail -n +2 hr_times.csv | while IFS= read -r line; do
            echo "${run},${line}" >> "$OUT_COMBINED"
        done

        # Optional: keep per-run CSVs
        mv hr_times.csv "$RUNS_DIR/hr_times_run_${t}_${run}.csv"
        mv hr_numbers.csv "$NUMBERS_DIR/hr_numbers_run_${t}_${run}.csv"
    done
done

echo "All runs completed. Combined results in $OUT_COMBINED"
echo "All per-run CSVs saved in folder: $RESULTS_DIR"
