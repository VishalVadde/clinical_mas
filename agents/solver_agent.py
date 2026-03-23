"""Solver Agent - LLaMA-3-8B fine-tuned on MIMIC-IV BHC pairs.

Training data: note-with-BHC-removed -> BHC text
This teaches the model clinical summarization using doctor-written gold-standard targets.
At inference, the system prompt controls whether output is clinical or patient-friendly.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME   = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = "/content/drive/MyDrive/clinical_mas/models/solver_checkpoints/final_adapter"

SYSTEM_PATIENT = (
    "You are a medical AI assistant. Read this clinical discharge note and write a "
    "3 to 5 sentence plain English summary for the patient. Focus on why they were "
    "admitted, what was done, their medications, and follow-up. Use simple words. "
    "Do not copy from the note."
)
SYSTEM_GENERAL = (
    "You are a clinical AI assistant. Write a concise clinical summary of this "
    "discharge note for a healthcare provider."
)


class SolverAgent:
    def __init__(self, hf_token=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb, device_map="auto", token=hf_token
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        self.model.eval()

    def summarize(self, note_text, role="patient"):
        system  = SYSTEM_PATIENT if role == "patient" else SYSTEM_GENERAL
        max_new = 200 if role == "patient" else 256
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Please summarize this discharge note:\n\n{note_text[:1500]}"},
        ]
        prompt    = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs    = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new, temperature=0.7, do_sample=True,
                repetition_penalty=1.3, no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
