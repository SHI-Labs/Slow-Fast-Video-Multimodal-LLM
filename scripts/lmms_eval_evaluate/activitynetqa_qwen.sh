MODEL_PATH=$1
TEST_FRAMES=$2

export HF_HOME=$(realpath ~/.cache/huggingface)

python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger_eng')"
python -m accelerate.commands.launch \
        --num_processes=8 \
        lmms_eval_evaluate.py \
        --model slowfast_videomllm \
        --model_args pretrained=${MODEL_PATH},video_decode_backend=pyav,conv_template=qwen_1_5,num_frames=${TEST_FRAMES},device_map='' \
        --tasks activitynetqa \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix activitynetqa_slowfastvideomllm_ \
        --output_path ./logs/  