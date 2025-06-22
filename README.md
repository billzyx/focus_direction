# Code for paper: Focus Directions Make Your Language Models Pay More Attention to Relevant Contexts

https://arxiv.org/abs/2503.23306

## Step:

1. Build environment

```
conda create --name focus_direction python=3.10 -y
conda activate focus_direction
pip install -r requirements.txt
# Note: make sure your transformers version is 4.45.2, since we change the model code (in edit_models) based on that version
```

2. Download data from "lost in the middel"

```
cd qa_data
wget https://nlp.stanford.edu/data/nfliu/lost-in-the-middle/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz
```

3. Generate dataset
```
python generate_focus_dataset.py
```

4. Obtain contextual heads

```
python get_contextual_heads.py
```

5. Obtain focus direction

```
bash run_train_kq_directions.sh
```

6. Use the focus direction to edit the model

```
bash apply_focus_direction.sh
```

7. Use the edited model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name_or_path = "edit_models_w/head_kq_directions/meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True
)
```

