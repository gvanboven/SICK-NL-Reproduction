import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sick_nl.code.models.bert_finetune import SICK_BERT_DATASET, BERTFineTuner


def load_bert_nli_model(sick_dataset, name, setting, train_data, dev_data, num_epochs,seed):
    print("Loading BERT model...")
    if setting == 'bert':
        tokenizer = BertTokenizer.from_pretrained(name)
        model = BertForSequenceClassification.from_pretrained(name, num_labels=3)
    elif setting == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(name)
        model = RobertaForSequenceClassification.from_pretrained(name, num_labels=3)
    print("Preparing datasets...")
    train_dataset = SICK_BERT_DATASET(train_data, tokenizer)
    eval_dataset = SICK_BERT_DATASET(dev_data, tokenizer)
    print("Loading finetuning model...")
    # below here I set num_epochs = num_epochs rather than num_epochs = 1, as previously the
    # model would only fine-tune for 1 epoch
    return BERTFineTuner(name, tokenizer, model, train_dataset, eval_dataset, seed,
                         num_epochs=num_epochs, freeze=False) #Changed this



