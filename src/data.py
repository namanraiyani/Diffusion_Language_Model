from datasets import load_dataset, load_from_disk, concatenate_datasets
from src.tokenizer import get_tokenizer


def prepare_pretraining_data(args):
    """
    prepare pretraining data from Gutenberg
    """
    context_length = args.context_length
    path_to_save = args.path_to_prepped_data
    cache_dir = args.huggingface_cache_dir

    print(f"Preparing pretraining data and saving to {path_to_save}")

    tokenizer = get_tokenizer(args.hf_model_name)

    if args.large_dataset:
        print("Loading large-scale datasets (FineWeb + Wikipedia)...")
        fw = load_dataset("HuggingFaceFW/fineweb", 
                        name="sample-10BT", 
                        split="train", 
                        cache_dir=cache_dir)
        
        fw_edu = load_dataset("HuggingFaceFW/fineweb-edu", 
                            name="sample-10BT", 
                            split="train", 
                            cache_dir=cache_dir)
        
        wiki = load_dataset("wikimedia/wikipedia", 
                            "20231101.en",
                            split="train",
                            cache_dir=cache_dir)
        
        fw = fw.remove_columns([col for col in fw.column_names if col != "text"])
        fw_edu = fw_edu.remove_columns([col for col in fw_edu.column_names if col != "text"])
        wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

        dataset = concatenate_datasets([fw, fw_edu, wiki])
    else:
        print("loading Project Gutenberg dataset ")
        dataset = load_dataset("manu/project_gutenberg", 
                               split="en",
                               cache_dir=cache_dir)
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
        
        if args.max_samples is not None:
            print(f"Original dataset size: {len(dataset)} books")
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            print(f"Using subset: {len(dataset)} books")

    dataset = dataset.train_test_split(test_size=args.test_split_pct, seed=args.dataset_split_seed)

    def compute_tokens(examples):
        tokenized = tokenizer(examples["text"], 
                              return_attention_mask=False, 
                              add_special_tokens=True,
                              max_length=None,
                              truncation=False)

        input_ids_list = []
        for ids in tokenized["input_ids"]:
            for i in range(0, len(ids), context_length):
                chunk = ids[i:i+context_length]
                if len(chunk) < context_length:
                    chunk = chunk + [tokenizer.pad_token_id] * (context_length - len(chunk))
                input_ids_list.append(chunk)
        
        return {"input_ids": input_ids_list}
    
    print("Tokenizing dataset (single process)...")
    tokenized_data = dataset.map(
        compute_tokens, 
        batched=True, 
        batch_size=args.batch_size,
        remove_columns="text"
    )

    print(f"Saving tokenized data to: {path_to_save}")
    tokenized_data.save_to_disk(path_to_save)

    return tokenized_data


def prepare_sft_data(args):
    context_length = args.context_length
    path_to_save = args.path_to_prepped_data
    cache_dir = args.huggingface_cache_dir

    print(f"Preparing SFT data and saving to {path_to_save}")

    tokenizer = get_tokenizer(args.hf_model_name)

    print("Loading OpenOrca dataset...")
    dataset = load_dataset("Open-Orca/OpenOrca", 
                           split="train", 
                           cache_dir=cache_dir)
    
    def apply_chat_template(query, response):
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ],
            tokenize=True,
            add_special_tokens=True,
        )
    
    def preprocess(example):
        instruction = example["question"]
        response = example["response"]
        
        tokenized = apply_chat_template(instruction, response)

        return {"input_ids": tokenized, "length": len(tokenized)}
    
    dataset = dataset.remove_columns([col for col in dataset.column_names 
                                     if col not in ["question", "response"]])
    
    dataset = dataset.train_test_split(test_size=args.test_split_pct, seed=args.dataset_split_seed)
    
    print("Tokenizing dataset ")
    tokenized_data = dataset.map(
        preprocess, 
        remove_columns=["question", "response"]
    )
    
    def keep_within_context(example):
        return example["length"] <= context_length
    
    print("Number of Samples In Dataset:", len(tokenized_data["train"]))
    tokenized_data = tokenized_data.filter(keep_within_context)
    tokenized_data = tokenized_data.remove_columns("length")
    print("Number of Samples After Length Filter:", len(tokenized_data["train"]))

    def get_answer_mask(example):
        tokenized = example["input_ids"]
        query_mask = []
        occurrence = 0
        is_answer = False
   
        for t in tokenized:
            check = (t == tokenizer.convert_tokens_to_ids("<END_ID>"))
            if not is_answer:
                query_mask.append(0)
            else:
                query_mask.append(1)

            if check:
                if occurrence == 0:
                    occurrence += 1
                else:
                    is_answer = True

        example["query_mask"] = query_mask
        return example

    print("Creating answer masks ")
    tokenized_data = tokenized_data.map(get_answer_mask)

    print(f"Saving tokenized data to: {path_to_save}")
    tokenized_data.save_to_disk(path_to_save)

    return tokenized_data
