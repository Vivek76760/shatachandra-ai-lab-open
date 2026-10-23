# Load order (Kaggle, accelerator GPU T4 x2, Internet on)

Run these cells one at a time, in order. Tested from a clean start on Kaggle (GPU T4 x2) on 23 October 2026 with patches v28r to v28z: a fact question, a calculation question and a Hindi long answer all worked. The two newest patches (v28aa and v28ab) were tested offline and in a working session, but not yet from a clean start. Other paths (English and Telugu long answers, file questions) were not re-tested from a clean start either, so please report any problem as an issue.

## 1. Get the files

```
!git clone https://github.com/Vivek76760/shatachandra-ai-lab-open
!cp shatachandra-ai-lab-open/*.py /kaggle/working/
```

## 2. Install

```
!pip install -q unsloth ddgs python-docx
```

## 3. Load the model

```
import contextlib, torch
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)
FastModel.for_inference(model)
model.disable_adapter = contextlib.nullcontext
```

## 4. Create an empty saved-answer file

```
open("/kaggle/working/kb.json", "w").write("{}")
```

## 5. Load the agent

```
code = open("/kaggle/working/agent_v28q_colab.py", encoding="utf-8").read()
exec(compile(code, "agent_v28q_colab.py", "exec"), globals())
```

## 6. Load the patches, in this order

The order matters. Do not sort the files by name.

```
patches = [
    "agent_v28r_calendar_patch.py", "agent_v28s_wiki_patch.py", "agent_v28t_latex_patch.py",
    "agent_v28u_caps_patch.py", "agent_v28v_warnfirst_patch.py", "agent_v28w_longreq_patch.py",
    "agent_v28x_hititle_patch.py", "agent_v28y_pron_patch.py", "agent_v28z_kbguard_patch.py",
    "agent_v28aa_notverified_patch.py", "agent_v28ab_leak_patch.py",
]
for p in patches:
    exec(compile(open("/kaggle/working/" + p, encoding="utf-8").read(), p, "exec"), globals())
```

## 7. Ask a question

```
print(agent_all("Who was the first Prime Minister of India?"))
```

Notes: answers are saved to /kaggle/working/kb.json. Do not publish that file. The first question takes longer because of web lookups.
