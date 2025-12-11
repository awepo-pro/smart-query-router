# run 

check `config.py` first. Make sure you have downloaded all the adapter/fine tuned models before run the program. To install model from huggingface, 

```
hf download MODLE_NAME
```

Lora models used in this project,
- SriSanthM/Qwen-1.5B-Tweet-Generations
- silent666/Qwen-Qwen2.5-1.5B-Instruct-1727478552
- DreamGallery/Qwen-Qwen2.5-1.5B-Instruct-1727452927
- Arthur-77/QWEN2.5-1.5B-medical-finetuned

for example, fine tuned mdoel `Arthur-77/QWEN2.5-1.5B-medical-finetuned`, 

```
hf download Arthur-77/QWEN2.5-1.5B-medical-finetuned
```

For adapter models, users have to specify the exact location in `config.py`, usually the downloaded place is `~/.cache/huggingface/hub`.

The main file is `main.py`. Majority of implementations are in `smart_qeury_router.py`. 
