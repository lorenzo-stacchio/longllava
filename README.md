conda create -n loongllava python=3.10 anaconda -y


conda activate loongllava


before installing with pip

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia


pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

## BUGS

flash attention version 2 issue

pip install flash-attn --no-build-isolation

## huggingface-cli login

huggingface-cli login 

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


# tirare giù e clonare le repo di long llava 9b e dentro ficcarci clip vit large

git clone https://huggingface.co/FreedomIntelligence/LongLLaVA-9B

cd LongLLaVA-9B

mkdir openai/

git clone https://huggingface.co/openai/clip-vit-large-patch14-336


# git 

apt-get install git-lfs

## dentro ogni cartella dei pesi presi da hugging-face (sia long llava che prima ha solo text e poi anche open ai clip)

git lfs install 

git lfs pull

## change safetenrso to ckpt

https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/8123#issuecomment-1445287087