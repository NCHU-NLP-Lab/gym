import time

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    pipeline
)

# Quantization configuration for memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Model and tokenizer setup
model_path = "meta-llama/Llama-3.2-1B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
processor.add_special_tokens({'pad_token': '<|PAD|>'})

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True
)
model.resize_token_embeddings(len(processor))

# Setup Accelerator
accelerator = Accelerator()

# Format prompt and questions
system_prompt = "You are a well qa system that can answer any question. Plsease answer the following questions:"    
questions = [
    "Why did the chicken cross the road?",
    "If cats could talk, what would they say to humans?",
    "Imagine a world where pizza grows on trees.",
    "What happens when you accidentally send a love letter to your boss?",
    "Describe the day a dog became the mayor of a small town.",
    "What if gravity stopped working every Tuesday afternoon?",
    "Tell me a bedtime story about a brave pancake.",
    "What would aliens think of human fashion trends?",
    "Write a hilarious guide on how to survive a zombie apocalypse with only a toaster.",
    "If unicorns had social media, what kind of posts would they share?",
] * 10  # 100 questions in total

formatted_questions = [
    processor.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q}
        ], 
        tokenize=False
    )
    for q in questions
]

# Setup pipeline
qa_pipeline = pipeline("text-generation", model=model, tokenizer=processor)

# Syncronize all GPUs
accelerator.wait_for_everyone()

start_time = time.time()

with accelerator.split_between_processes(formatted_questions, apply_padding=True) as questions:
    results = {"num_tokens": 0}
    for question in formatted_questions:
        answer = qa_pipeline(question, max_length=256, do_sample=True, temperature=0.9)
        answer = answer[0]["generated_text"]
        if "<|eot_id|>assistant" in answer:
            answer = answer.split("<|eot_id|>assistant")[-1].strip()
        else:
            answer = answer.strip()
            
        results["num_tokens"] += len(processor.tokenize(answer))
        
    results = [results]
    
results = gather_object(results)

if accelerator.is_main_process:
    elapsed_time = time.time() - start_time
    total_tokens = sum([r["num_tokens"] for r in results])
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Token per second: {total_tokens / elapsed_time:.2f}")

if dist.is_initialized():
    dist.destroy_process_group()
    
# Elapsed time: 175.44 seconds
# Token per second: 208.93