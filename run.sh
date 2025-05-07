#!/bin/bash
# example: bash ./run.sh model_name="KAN" seq_len=12 pre_len=6

model_name="KAN"
seq_len=12
pre_len=6

result_dir="./results"
if [ ! -d "$result_dir" ]; then
    mkdir -p "$result_dir" && echo "make dictionary $result_dir"
fi

data_dir="./data"
if [ ! -d "$data_dir" ]; then
    mkdir -p "$data_dir" && echo "make dictionary $data_dir"
fi

checkpoints_dir="./checkpoints"
if [ ! -d "$checkpoints_dir" ]; then
    mkdir -p "$checkpoints_dir" && echo "make dictionary $checkpoints_dir"
fi

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
    pre_len=*)
      IFS=',' read -ra pre_len_arr <<< "${ARG#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $ARG"
      exit 1
      ;;
  esac
done

model_list=("KAN" "Wavelet")

if [ ${#pre_len_arr[@]} -eq 0 ]; then
    pre_len_arr=("$pre_len")
fi

for p in "${pre_len_arr[@]}"; do
  if [ "$model_name" = "ALL" ]; then
      for mod in "${model_list[@]}"; do
          python -u start.py \
          --model_name="$mod" \
          --seq_len="$seq_len" \
          --pre_len="$p"
      done
  else
      python -u start.py \
      --model_name="$model_name" \
      --seq_len="$seq_len" \
      --pre_len="$p"
  fi
done

if [ $? -eq 0 ]; then
    echo "Run finished!"
else
    echo "Run failed."
fi