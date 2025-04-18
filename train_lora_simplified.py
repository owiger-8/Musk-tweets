def train_lora_model_simplified(base_model_path, dataset, output_dir, role):
    """Train a LoRA model using the default Trainer"""
    
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
    
    # Tokenize dataset
    def tokenize_function(examples):
        # Combine prompt and response
        texts = [prompt + " " + response for prompt, response in zip(examples["prompt"], examples["response"])]
        
        # Tokenize
        tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        
        # Create labels (same as input_ids for causal language modeling)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Create trainer
    from transformers import Trainer
    
    trainer = Trainer(
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