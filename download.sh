export HF_ENDPOINT="https://hf-mirror.com"

token=YOUR_HUGGINGFACE_TOKEN

LOCAL_DIR="./ckpts"
mkdir -p $LOCAL_DIR


remote_name="Qwen/Qwen2.5-0.5B-Instruct"
model_name="Qwen2.5-0.5B-Instruct"
huggingface-cli download $remote_name --local-dir $LOCAL_DIR/$model_name --token $token


remote_name="Qwen/Qwen2.5-1.5B-Instruct"
model_name="Qwen2.5-1.5B-Instruct"
huggingface-cli download $remote_name --local-dir $LOCAL_DIR/$model_name --token $token



remote_name="Qwen/Qwen2.5-3B-Instruct"
model_name="Qwen2.5-3B-Instruct"
huggingface-cli download $remote_name --local-dir $LOCAL_DIR/$model_name --token $token


remote_name="Qwen/Qwen-7B-Chat"
model_name="Qwen-7B-Chat"
huggingface-cli download $remote_name --local-dir $LOCAL_DIR/$model_name --token $token



remote_name="Qwen/Qwen-14B-Chat"
model_name="Qwen-14B-Chat"
huggingface-cli download $remote_name --local-dir $LOCAL_DIR/$model_name --token $token



remote_name="Qwen/Qwen-72B-Chat"
model_name="Qwen-72B-Chat"
huggingface-cli download $remote_name --local-dir $LOCAL_DIR/$model_name --token $token


remote_name="baichuan-inc/Baichuan2-13B-Chat"
model_name="Baichuan2-13B-Chat"
huggingface-cli download $remote_name --local-dir $LOCAL_DIR/$model_name --token $token


remote_name="meta-llama/Llama-3.1-8B-Instruct"
model_name="LLaMA-3.1-8B-Instruct"
huggingface-cli download $remote_name --local-dir $LOCAL_DIR/$model_name --token $token --exclude "original/*"


remote_name="THUDM/glm-4-9b-chat"
model_name="GLM4-9B-Chat"
huggingface-cli download $remote_name --local-dir $LOCAL_DIR/$model_name --token $token