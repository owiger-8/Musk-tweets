import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import os

# Sample dataset
sample_data = [
    {
        "input": "Just bought $10M worth of DogeCoin. The future of currency is here! 🚀",
        "title": "Elon Musk's DogeCoin Purchase Sends Crypto Markets Into Frenzy",
        "stock": "DOGE-USD",
        "meme": "Elon holding a literal doge rocket to the moon (50k upvotes)",
        "report": "Musk's influence on crypto remains volatile. DOGE-USD may see short-term surge but lacks fundamentals.",
        "price": 0.45,
        "output": [0.52, 0.61, 0.58, 0.49, 0.42]
    },
    {
        "input": "No more orders from China for imported Tesla models. Focusing on US production.",
        "title": "Tesla Stops Taking New Orders in China for Two Imported, US-Made Models",
        "stock": "TSLA",
        "meme": "When China bans Tesla for no reason: looks like I'll just have to buy China then",
        "report": "Tesla's China strategy shift may impact short-term revenue but strengthen long-term US manufacturing position.",
        "price": 241.38,
        "output": [235.50, 230.20, 233.80, 245.60, 250.30]
    }
]

def prepare_dataset(data, role):
    """Prepare dataset for training based on role"""
    
    if role == "cnn":
        # Prepare CNN headline dataset
        prompts = []
        responses = []
        
        for item in data:
            prompt = f"You are a CNN news headline writer. Write a headline for a news article based on this tweet from Elon Musk:\n\nTweet: {item['input']}\n\nCNN Headline:"
            response = item['title']
            prompts.append(prompt)
            responses.append(response)
            
    elif role == "reddit":
        # Prepare Reddit meme dataset
        prompts = []
        responses = []
        
        for item in data:
            prompt = f"You are a Reddit user creating a viral meme title. Create a funny meme title based on this tweet from Elon Musk and the CNN headline:\n\nTweet: {item['input']}\nCNN Headline: {item['title']}\n\nReddit Meme Title:"
            response = item['meme']
            prompts.append(prompt)
            responses.append(response)
            
    elif role == "goldman":
        # Prepare Goldman Sachs report dataset
        prompts = []
        responses = []
        
        for item in data:
            prompt = f"You are a Goldman Sachs analyst writing a report. Write a brief analysis based on this tweet from Elon Musk:\n\nTweet: {item['input']}\n\nGoldman Sachs Analysis:"
            response = item['report']
            prompts.append(prompt)
            responses.append(response)
            
    elif role == "market":
        # Prepare market prediction dataset
        prompts = []
        responses = []
        
        for item in data:
            prompt = f"You are a market prediction AI. Predict the stock price movement for the next 5 days based on this information:\n\nTweet from Elon Musk: {item['input']}\nCNN Headline: {item['title']}\nAnalyst Report: {item['report']}\nCurrent Price: ${item['price']}\n\nPredict the stock price for the next 5 days as a comma-separated list of values:"
            response = ",".join([str(x) for x in item['output']])
            prompts.append(prompt)
            responses.append(response)
    
    # Create dataset
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "response": responses
    })
    
    return dataset

def train_lora_model(base_model_path, dataset, output_dir, role):
    """Train a LoRA model"""
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch"
    )
    
    # Define trainer
    from transformers import Trainer
    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Get inputs
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            labels = inputs["labels"]
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Return loss
            return (outputs.loss, outputs) if return_outputs else outputs.loss
    
    # Tokenize dataset
    def tokenize_function(examples):
        # Tokenize prompts and responses
        prompt_tokens = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)
        response_tokens = tokenizer(examples["response"], truncation=True, padding="max_length", max_length=128)
        
        # Create input_ids by concatenating prompt and response
        input_ids = []
        attention_mask = []
        labels = []
        
        for p_ids, p_mask, r_ids in zip(prompt_tokens["input_ids"], prompt_tokens["attention_mask"], response_tokens["input_ids"]):
            # Combine prompt and response
            combined_ids = p_ids + r_ids[1:]  # Skip the BOS token
            combined_mask = p_mask + [1] * (len(r_ids) - 1)
            
            # Create labels (-100 for prompt tokens, actual ids for response tokens)
            combined_labels = [-100] * len(p_ids) + r_ids[1:]
            
            input_ids.append(combined_ids)
            attention_mask.append(combined_mask)
            labels.append(combined_labels)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train model
    trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model for {role} trained and saved to {output_dir}")

def main():
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/cnn", exist_ok=True)
    os.makedirs("models/reddit", exist_ok=True)
    os.makedirs("models/goldman", exist_ok=True)
    os.makedirs("models/market", exist_ok=True)
    '''
    # Save sample data to file
    with open("dataset.json", "w") as f:
        json.dump(sample_data, f, indent=2)'''
    
    # Base model path
    base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Train CNN model
    cnn_dataset = prepare_dataset(sample_data, "cnn")
    train_lora_model(base_model_path, cnn_dataset, "models/cnn", "cnn")
    
    # Train Reddit model
    reddit_dataset = prepare_dataset(sample_data, "reddit")
    train_lora_model(base_model_path, reddit_dataset, "models/reddit", "reddit")
    
    # Train Goldman Sachs model
    goldman_dataset = prepare_dataset(sample_data, "goldman")
    train_lora_model(base_model_path, goldman_dataset, "models/goldman", "goldman")
    
    # Train Market model
    market_dataset = prepare_dataset(sample_data, "market")
    train_lora_model(base_model_path, market_dataset, "models/market", "market")

if __name__ == "__main__":
    main()
