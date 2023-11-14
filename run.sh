SQUAD_DIR="./data/squad_v2/raw/" 
MODEL_DIR="./bert_fine_tuned_model/checkpoint-250000"

python3 run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --version_2_with_negative \
    --save_steps 20000 \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --per_gpu_train_batch_size 1 \
    --num_train_epochs 2 \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --local_rank -1 \
    --output_dir bert_fine_tuned_model_2 \
    --overwrite_output_dir \
    --version_2_with_negative \
    # --evaluate_during_training \
    # --eval_all_checkpoints \
    # --max_steps 1500 \
    # --overwrite_cache
    #    --do_train \