import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def load_lora_model(base_model_path, lora_path):
    """
    Load a LoRA fine-tuned model
    
    Args:
        base_model_path: Path to the base TinyLlama model
        lora_path: Path to the LoRA weights
        
    Returns:
        model: The loaded model with LoRA weights
        tokenizer: The tokenizer for the model
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA weights
    if os.path.exists(lora_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    """
    Generate text using the model
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: The input prompt
        max_length: Maximum length of generated text
        
    Returns:
        generated_text: The generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the generated text
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

def prepare_cnn_prompt(tweet):
    """Prepare prompt for CNN headline generation"""
    return f"""
    You are a CNN news headline writer. Write a headline for a news article based on this tweet from Elon Musk:
    
    Tweet: {tweet}
    
    CNN Headline:
    """

def prepare_reddit_prompt(tweet, cnn_headline):
    """Prepare prompt for Reddit meme generation"""
    return f"""
    You are a Reddit user creating a viral meme title. Create a funny meme title based on this tweet from Elon Musk and the CNN headline:
    
    Tweet: {tweet}
    CNN Headline: {cnn_headline}
    
    Reddit Meme Title:
    """

def prepare_goldman_prompt(tweet):
    """Prepare prompt for Goldman Sachs report generation"""
    return f"""
    You are a Goldman Sachs analyst writing a report. Write a brief analysis based on this tweet from Elon Musk:
    
    Tweet: {tweet}
    
    Goldman Sachs Analysis:
    """

def prepare_market_prompt(tweet, cnn_headline, goldman_report, current_price):
    """Prepare prompt for market prediction"""
    return f"""
    You are a market prediction AI. Predict the stock price movement for the next 5 days based on this information:
    
    Tweet from Elon Musk: {tweet}
    CNN Headline: {cnn_headline}
    Analyst Report: {goldman_report}
    Current Price: ${current_price}
    
    Predict the stock price for the next 5 days as a comma-separated list of values:
    """