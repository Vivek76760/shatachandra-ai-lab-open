# Load order (Kaggle, accelerator GPU T4 x2, Internet on)

Copy all the .py files from this folder into /kaggle/working, then run these cells one at a time.

1. Install:

    !pip install -q unsloth ddgs python-docx

2. Load the model:

    import contextlib, torch
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained("unsloth/gemma-4-E4B-it-unsloth-bnb-4bit", max_seq_length=4096, load_in_4bit=True)
    FastModel.for_inference(model)
    model.disable_adapter = contextlib.nullcontext

3. Create an empty saved-answer file:

    open("/kaggle/working/kb.json", "w").write("{}")

4. Load the agent:

    exec(compile(open("/kaggle/working/agent_v28q_colab.py", encoding="utf-8").read(), "agent", "exec"), globals())

5. Load the patches, in this order: r, s, t, u, v, w, x, y, z:

    for p in ["agent_v28r_calendar_patch.py", "agent_v28s_wiki_patch.py", "agent_v28t_latex_patch.py", "agent_v28u_caps_patch.py", "agent_v28v_warnfirst_patch.py", "agent_v28w_longreq_patch.py", "agent_v28x_hititle_patch.py", "agent_v28y_pron_patch.py", "agent_v28z_kbguard_patch.py"]:
        exec(compile(open("/kaggle/working/" + p, encoding="utf-8").read(), p, "exec"), globals())

6. Ask a question:

    print(agent_all("Who was the first Prime Minister of India?"))

This guide has not yet been tested from a clean start.
