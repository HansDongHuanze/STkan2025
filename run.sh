#!/bin/bash
# example: bash ./run.sh model_name="KAN" seq_len=12

model_name="KAN"
seq_len=12

for ARG in "$@"; do
  case $ARG in
    model_name=*)
      model_name="${ARG#*=}"
      shift
      ;;
    seq_len=*)
      seq_len="${ARG#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $ARG"
      exit 1
      ;;
  esac
done

model_list=("KAN" "Wavelet")

if [ "$model_name" = "ALL" ]; then
    for mod in "${model_list[@]}"; do
        python -u start.py \
        --model_name="$mod" \
        --seq_len="$seq_len"
    done
else
    python -u start.py \
    --model_name="$model_name" \
    --seq_len="$seq_len"
fi

if [ $? -eq 0 ]; then
    echo "Run finished!"
else
    echo "Run failed."
fi