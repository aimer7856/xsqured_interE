#!/bin/bash

# === Usage: ./log_array_seff.sh 45341166 seff_output.txt ====

ARRAY_JOBID=$1
OUTPUT_FILE=$2

if [[ -z "$ARRAY_JOBID" || -z "$OUTPUT_FILE" ]]; then
  echo "Usage: $0 <array_jobid> <output_filename>"
  exit 1
fi

# Clear output file first
echo "seff output for array job $ARRAY_JOBID" > "$OUTPUT_FILE"
echo "=======================================" >> "$OUTPUT_FILE"

# Get valid array task IDs only (e.g., 45341166_0, 45341166_1, ...)
mapfile -t JOB_IDS < <(sacct -j "$ARRAY_JOBID" --format=JobID --noheader | awk '{print $1}' | grep -E "^${ARRAY_JOBID}_[0-9]+$")

for JOBID in "${JOB_IDS[@]}"; do
  echo "---------------------------------------" >> "$OUTPUT_FILE"
  echo "Job ID: $JOBID" >> "$OUTPUT_FILE"
  seff "$JOBID" >> "$OUTPUT_FILE" 2>&1
  echo "" >> "$OUTPUT_FILE"
done

echo "Done. Output saved to $OUTPUT_FILE"