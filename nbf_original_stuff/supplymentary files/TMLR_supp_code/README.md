# Supplementary Material for TMLR submission

## Install
Create a Conda environment and install the required packages.
```
conda create -n nbf-llm python=3.10
conda activate nbf-llm
pip install -r requirements.txt
```

Please set up the LLM API keys in your environment. Create a `.env` file in the root path with the following content.

```
BASE_URL_GPT="https://api.openai.com/v1"
GPT_API_KEY="XXX"

BASE_URL_LLAMA="https://api.llama-api.com"
LLAMA_API_KEY="XXX"

CLAUDE_API_KEY="XXX"
```

## Data preparation
To collect data of ActorAttack, following the instruction of [ActorAttack](https://github.com/AI45Lab/ActorAttack), collect the multi-turn attack results based on single-turn  queries of [1k Circuit Breakers queries](https://github.com/AI45Lab/ActorAttack/blob/master/data/circuit_breaker_train.csv) under GPT-3.5-turbo. Move the results to `./data/train/actorattack_1k_cb.json`. Similaly, collect multi-turn attack results based on single-turn [200 Harmbench queries](https://github.com/AI45Lab/ActorAttack/blob/master/data/harmbench.csv) and save the result `./data/val/actorattack_200_hb.json` as validation data.
Regarding attacks of Cressendo, Acronym, Opposite-day, following the instructions in this [repo](https://github.com/AIM-Intelligence/Automated-Multi-Turn-Jailbreaks), collect each multi-turn attack result as training data from 1k Circuit Breakers queries and as validation data from 200 Harmbench queries. Save the training files in `./data/train/XXX.jsonl` and the val files in `./data/val/XXX.jsonl`.

## NBF Model training
Run the following script to train neural dialogue dynamics and barrier function based on the multi-turn attack data collected. 

```
python train.py --save_path "./models" --find_embedding
```

## NBF Model evaluation
To evaluate the safety steering against multi-turn jailbreaks, run the following script to find results without steering for victim LLMs of `--target_model` against different attacks of `--attack_method`, which are listed in `./attacks`. All API-based LLMs are supported in `--target_model` once API keys are set up.

```
python steering.py --attack_method "opposite_day"  --target_model "gpt-3.5-turbo"
```

Run the safety steering with pre-trained neural dynamics and barrier function at `./models/XXX.pth` and threshold of 0.001 using the following script,

```
python steering.py --attack_method "opposite_day"  --target_model "gpt-3.5-turbo" --add_safety_index --safety_filtering --model_path "./models/XXX.pth" --threshold 0.001
```

Besides, open-source and LoRA fine-tuned LLMs are also supported. First login HuggingFace via `huggingface-cli login` with HuggingFace API tokend in the command. Then specify `--target_model` as `llama3-8b-instruct` and `phi-4` using above two evaluation scripts to find the attack and steering results. Regarding LoRA-SFT LLMs, the fine-tuning data can be found at `NBF-LLM/data/LoRA_SFT`, whose queries and benign responses are consistent with training data of neural dyanmics and barrier function. Follow the instruction of [LLAMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for SFT with LoRA and replace `PATH_TO_FINETUNED_LLAMA3` and `PATH_TO_FINETUNED_PHI4` in `.attacks/utils/generate.py` with the paths of the fine-tuned models. Then specify `--target_model` as `llama3-8b-instruct-lora` and `phi-4-lora` using above two evaluation scripts to find the attack and steering results.


