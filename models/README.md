# models/

After running `python train.py`, the fine-tuned BERT model will be saved here under `bert_resume/`:

```
bert_resume/
├── config.json
├── pytorch_model.bin   (or model.safetensors)
├── tokenizer_config.json
├── vocab.txt
├── special_tokens_map.json
└── label_map.json      (auto-generated: index → class name)
```

The Streamlit app (`app.py`) and `predict.py` will automatically load from this directory.
