from datasets import load_dataset
from pip._internal.cli.cmdoptions import cache_dir
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoTokenizer
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
import os
import json
import torch

def run_pretrain():
    # print(torch.cuda.is_available())
    script_path = os.path.realpath(__file__).replace('\\', '/').replace('/train_from_scratch.py', '')
    training_path = script_path + "/dataset/"
    training_files = [training_path + file for file in os.listdir(training_path)]

    model_path = script_path + "/model_files/"
    tokenizer_path = script_path + "/tokenizer_files/"

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)

    # save the training set to train.txt
    special_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
    ]
    # 30,522 vocab is BERT's default vocab size, feel free to tweak
    vocab_size = 30522
    # maximum sequence length, lowering will result to faster training (when increasing batch size)
    max_length = 512
    # whether to truncate
    truncate_longer_samples = True
    def train_tokenizer(files, tokenizer_output_dir, vocab_size=30522, max_length=512, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"], truncate_longer_samples=True):
        # initialize the WordPiece tokenizer
        tokenizer = BertWordPieceTokenizer()
        # train the tokenizer
        tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
        # enable truncation up to the maximum 512 tokens
        tokenizer.enable_truncation(max_length=max_length)
        # make the directory if not already there
        if not os.path.isdir(tokenizer_output_dir):
            os.mkdir(tokenizer_output_dir)
        # save the tokenizer
        tokenizer.save_model(tokenizer_output_dir)
        # dumping some of the tokenizer config to config file,
        # including special tokens, whether to lower case and the maximum sequence length
        with open(os.path.join(tokenizer_output_dir, "config.json"), "w") as f:
            tokenizer_cfg = {
                "do_lower_case": True,
                "unk_token": "[UNK]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
                "model_max_length": max_length,
                "max_len": max_length,
            }
            json.dump(tokenizer_cfg, f)
    # train_tokenizer(files=training_files, tokenizer_output_dir=tokenizer_path)
    print("loading dataset")
    dataset = load_dataset("text", data_files=training_files, split='train', cache_dir=script_path+"/cache/")
    print("loaded")
    print(dataset)
    d = dataset.train_test_split(test_size=0.1)
    def dataset_to_text(dataset, output_filename="data.txt"):
    """Utility function to save dataset text to disk,
    useful for using the texts to train the tokenizer
    (as the tokenizer accepts files)"""
    with open(output_filename, "w") as f:
        for t in dataset["text"]:
        print(t, file=f)

    # save the training set to train.txt
    dataset_to_text(d["train"], "train.txt")
    # save the testing set to test.txt
    dataset_to_text(d["test"], "test.txt")

    # tokenizer=BertTokenizer(vocab_file=model_path+'vocab.json')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def encode_with_truncation(examples):
        """Mapping function to tokenize the sentences passed with truncation"""
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)

    def encode_without_truncation(examples):
        """Mapping function to tokenize the sentences passed without truncation"""
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    # the encode function will depend on the truncate_longer_samples variable
    encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

    # tokenizing the train dataset
    train_dataset = d["train"].map(encode, batched=True)
    # tokenizing the testing dataset
    test_dataset = d["test"].map(encode, batched=True)
    if truncate_longer_samples:
        # remove other columns and set input_ids and attention_mask as
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    else:
        test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
        train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        return result
    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    if not truncate_longer_samples:
        train_dataset = train_dataset.map(group_texts, batched=True, batch_size=2_000,
                                        desc=f"Grouping texts in chunks of {max_length}")
        test_dataset = test_dataset.map(group_texts, batched=True, batch_size=2_000,
                                    num_proc=4, desc=f"Grouping texts in chunks of {max_length}")

    # initialize the model with the config
    model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
    model = BertForMaskedLM(config=model_config)

    # initialize the data collator, randomly masking 20% (default is 15%) of the tokens for the Masked Language
    # Modeling (MLM) task
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )

    training_args = TrainingArguments(
        output_dir=model_path,          # output directory to where save model checkpoint
        evaluation_strategy="steps",    # evaluate each `logging_steps` steps
        overwrite_output_dir=True,
        num_train_epochs=60,            # number of training epochs, feel free to tweak
        per_device_train_batch_size=8, # the training batch size, put it as high as your GPU memory fits
        gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
        per_device_eval_batch_size=10,  # evaluation batch size
        logging_steps=500,             # evaluate, log and save model checkpoints every 1000 step
        save_steps=500,
        load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
        save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
        learning_rate=5e-6,           # the learning rate
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # train the model
    trainer.train()
    trainer.save_model(model_path)

if __name__ == "__main__":
    run_pretrain()