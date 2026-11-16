import torch

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

app = FastAPI(
    title="Priyangan AI - Local LLM API",
    description="Local GPU-powered language model API",
    version="1.0"
)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Configure 4-bit quantisation
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model
print("Loading Qwen model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto"
)

model.eval()

print("LLM successfully loaded!")
print("GPU:", torch.cuda.get_device_name(0))


# Request structure
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 150


# Home endpoint
@app.get("/")
def home():
    return {
        "status": "Online",
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0)
    }


# AI generation endpoint
@app.post("/generate")
def generate(request: ChatRequest):

    messages = [
        {"role": "user", "content": request.prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return {
        "model": MODEL_NAME,
        "response": response
    }