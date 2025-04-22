

## Initialization:
```
srun --cpus-per-task=4 --mem=100GB --time=12:00:00 --gres=gpu:rtx8000:4 --pty /bin/bash
<!-- conda create -n llm_pvc python=3.10 -->
conda activate llm_pvc
cd WorkingCode
pip install -r requirements.txt
git submodule update --init --recursive
jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888
```


```
ssh-keygen -t rsa -C "yunweizhao18@gmail.com"
cat ~/.ssh/id_rsa.pub
cd /workspace/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# /workspace/miniconda3
echo 'export PATH="/workspace/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda create -n llm python=3.10
conda init
conda activate llm
cd /workspace/work/Contextual-Integrity/WorkingCode/LS-LLaMA
pip install -r requirements.txt
pip install llama-recipes fastcore "transformers!=4.38.*,!=4.39.*" --extra-index-url https://download.pytorch.org/whl/test/cuda127
pip install bitsandbytes>=0.43.0
pip install wandb
pip install python-dotenv
apt update; apt install vim
pip install seqeval
cd /workspace/work/Contextual-Integrity/WorkingCode/LS-LLaMA
pip install -r requirements.txt
echo 'export TRANSFORMERS_CACHE=/workspace/work/cache/' >> ~/.bashrc
echo 'export HF_HOME=/workspace/work/cache/' >> ~/.bashrc
echo 'export XDG_CACHE_HOME=/workspace/work/cache/' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/workspace/work/cache/' >> ~/.bashrc
source ~/.bashrc
conda activate llm
huggingface-cli login --token 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
```
```
echo 'export TRANSFORMERS_CACHE=/scratch/yz5944/nlp+privacy/Contextual-Integrity/cache/' >> ~/.bashrc
echo 'export HF_HOME=/scratch/yz5944/nlp+privacy/Contextual-Integrity/cache/' >> ~/.bashrc
echo 'export XDG_CACHE_HOME=/scratch/yz5944/nlp+privacy/Contextual-Integrity/cache/' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/scratch/yz5944/nlp+privacy/Contextual-Integrity/cache/' >> ~/.bashrc
```

to adapt to old code:
1. replace the indices line of code in llama_modeling
2. replace 1.1.1 to 0.34.2 version -> accelerator

remove module: git rm -r --cached xxx
sync: b2 sync Contextual-Integrity b2://llmpvc2/Contextual-Integrity --skip-newer
b2 sync b2://llmpvc2/Contextual-Integrity Contextual-Integrity --skip-newer