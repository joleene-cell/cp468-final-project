import os
import time
import json
from tqdm import tqdm
import ollama
from preprocess import prepare_multi30k_loaders

MODEL_NAME = "llama3.1:8b"

# --- Prompt Definitions ---
# Variant 1: Direct and simple instruction
PROMPT_V1_SYS = "You are a machine translation system. Translate the following English sentence to German. Output only the translation."

# Variant 2: Persona-based with linguistic focus
PROMPT_V2_SYS = "You are an expert bilingual linguist specializing in English-to-German translation. Provide a natural, fluent, and grammatically precise German translation for the provided English text. Do not include any explanations or alternative options."

# Few-shot examples (k=3) taken from standard translation datasets
FEW_SHOT_EXAMPLES = [
    {"en": "A man in an orange hat starring at something.", "de": "Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt."},
    {"en": "A Boston Terrier is running on lush green grass in front of a white fence.", "de": "Ein Boston Terrier läuft auf saftig-grünem Gras vor einem weißen Zaun."},
    {"en": "A girl in karate uniform breaking a stick with a front kick.", "de": "Ein Mädchen in einem Karateanzug bricht einen Stock mit einem Tritt."}
]

def format_prompt(english_text: str, variant: int, k: int) -> list:
    """Formats the messages array for the Chat API based on variant and k-shots."""
    messages = []
    sys_msg = PROMPT_V1_SYS if variant == 1 else PROMPT_V2_SYS
    messages.append({"role": "system", "content": sys_msg})
    
    if k > 0:
        for ex in FEW_SHOT_EXAMPLES[:k]:
            messages.append({"role": "user", "content": ex["en"]})
            messages.append({"role": "assistant", "content": ex["de"]})
            
    messages.append({"role": "user", "content": english_text})
    return messages

def call_llm(messages: list, model: str = MODEL_NAME):
    """Calls the locally-running Ollama model. No rate limits to back off from
    since this isn't a shared API -- retries aren't needed the same way."""
    start_time = time.time()
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0.1, "num_predict": 60},  # num_predict mirrors max_tokens
    )
    latency = time.time() - start_time

    translation = response["message"]["content"].strip()
    prompt_tokens = response.get("prompt_eval_count", 0)
    completion_tokens = response.get("eval_count", 0)
    return translation, latency, prompt_tokens, completion_tokens

def run_evaluation(test_data: list, model_name: str = MODEL_NAME):
    """Runs the full evaluation suite across prompt variants and shot configurations."""
    results = []
    
    # Configurations: Variant 1 (0-shot, 3-shot), Variant 2 (0-shot, 3-shot)
    configs = [
        {"variant": 1, "k": 0},
        {"variant": 1, "k": 3},
        {"variant": 2, "k": 0},
        {"variant": 2, "k": 3}
    ]
    
    for config in configs:
        print(f"\nRunning Configuration: Variant {config['variant']}, {config['k']}-shot")
        total_latency, total_prompt_tokens, total_comp_tokens = 0, 0, 0
        config_predictions = []
        
        for idx, text in enumerate(tqdm(test_data)):
            # Reconstruct original English sentence from preprocessed tokens
            eng_sentence = " ".join(text) 
            messages = format_prompt(eng_sentence, config['variant'], config['k'])
            
            try:
                translation, latency, p_tokens, c_tokens = call_llm(messages, model_name)
                total_latency += latency
                total_prompt_tokens += p_tokens
                total_comp_tokens += c_tokens
                
                config_predictions.append({
                    "id": idx,
                    "source": eng_sentence,
                    "prediction": translation
                })
            except Exception as e:
                print(f"Failed on sample {idx}: {e}")
                
        # Log aggregated stats
        summary = {
            "configuration": f"V{config['variant']}_{config['k']}shot",
            "total_requests": len(test_data),
            "total_time_seconds": total_latency,
            "avg_latency_seconds": total_latency / len(test_data) if test_data else 0,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_comp_tokens,
            "predictions": config_predictions
        }
        results.append(summary)
        
        # Save checkpoints incrementally
        with open(f"llm_results_v{config['variant']}_{config['k']}shot.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

    return results

if __name__ == "__main__":
    print("Loading Multi30k test data...")
    # Utilize the existing preprocessing script to fetch the test set
    _, _, _, _, _, raw_test_src, _ = prepare_multi30k_loaders()
    
    print(f"Loaded {len(raw_test_src)} test samples.")
    run_evaluation(raw_test_src)
    print("LLM Baseline evaluation complete. Results saved to JSON.")