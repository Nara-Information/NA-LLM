"""causal.py
version: 1.0.1

Fine-tuning with Seq2Seq objective with BART-like models.

Version history
1.0 model is trainable with the script
1.0.1
  - implemented yaml-based configurations 
  - made a module out of the script

Author: Gyu-min Lee
his.nigel at gmail dot com
"""

import json 
import csv
import random

from datetime import datetime 

import yaml
import torch
import wandb
import evaluate 

from tqdm import tqdm
from transformers import AutoTokenizer as Tokenizer
from transformers import AutoModelForSeq2SeqLM as Model
from transformers import GenerationConfig
from torch.utils.data import Dataset, DataLoader 

device = 'cuda' if torch.cuda.is_available() else \
             'mps' if torch.backends.mps.is_available() else 'cpu'

def _parseConfig(configPath) -> dict:
    """
    Parse a `yaml` config.

    For streamlined processing, it is desirable to parse the master 
    `config.yaml` and pass the config around from `main.py`.
    However, `_parseConfig()` is included as well to ensure
    portability of the file.
    Thus, `configPath` here must include at least all the configurations
    required for this script.
    
    Returns:
        dict: dictionary of configs.
    """

    with open(configPath) as f:
        config = yaml.load(f, yaml.Loader)
    
    return config
    
def assert_configLegality(config: dict):
    """
    Test the legality of the config -- helps failing fast.

    Runs several assertions to ensure that the required configs 
    are available. The routine will also check for basic legality
    check for the configs.

    Note that the test is script specific. Call other module's 
    `testConfig` to ensure the config will work there. Particularly,
    from the master config, the routine will check for configs
    that should be found under 'training'.

    Params:
        config (dict): config to test. If it was from master config,
            pass 'training' configs only.

    Returns:
        nothing. However, if any of the assertion occurs, 
            AssertionError is raised. If no AssertionError was 
            raised, it is probably good to go.
            See the AssertionError message to check which 
            configurations are missing.
    """

    assert 'trainingTranslateLM' in config
    translateConfig = config['trainingTranslateLM']
    assert 'modelIdentifier' in translateConfig 
    assert 'dataPath' in translateConfig
    assert 'saveLocal' in translateConfig
    if translateConfig['saveLocal']:
        assert 'modelSavePath' in translateConfig, \
            "modelSavePath must be specified if saveLocal is set."
    assert 'saveToHF' in translateConfig
    if translateConfig['saveToHF']:
        assert 'modelSaveIdentifier' in translateConfig, \
            "modelSaveIdentifier must be specified if saveToHF is set."
    assert 'maxEpoch' in translateConfig
    assert 'trainBatch' in translateConfig 
    assert type(translateConfig['trainBatch']) == int, \
        "batch size should be an integer."
    if 'evalBatch' in translateConfig:
        assert type(translateConfig['evalBatch']) == int, \
            "batch size should be an integer."
    assert 'evalWith' in translateConfig
    evalWith = translateConfig['evalWith']
    if type(evalWith) != list:
        evalWith = [evalWith]
    assert False not in [True if e.lower() in ['bleu', 'sts', 'none'] else False for e in evalWith], \
        f"evalWith must be a combination of 'BLEU', 'sts', or 'none' but got {evalWith}"
    if 'sts' in evalWith:
        assert 'stsModelIdentifier' in translateConfig
    if 'bleu' in evalWith:
        assert 'bleuMethod' in translateConfig
        assert translateConfig['bleuMethod'] in ['sacrebleu', 'custom'], \
            "bleuMethod must be 'sacrebleu' or custom"
        if translateConfig['bleuMethod'] == 'custom':
            assert 'bleuScript' in translateConfig, \
                "The script file defining the bleu score should be specified."
    assert 'reportToWandb' in translateConfig
    if translateConfig['reportToWandb']:
        assert 'wandbProjectName' in translateConfig
        assert 'wandbRunName' in translateConfig 
        if 'wandbConfig' in translateConfig:
            assert type(translateConfig['wandbConfig']) == dict, \
                    f"'wandbConfig' expected to be of type dict but got {type(translateConfig['wandbConfig'])}"
    return

class InqueryDataset(Dataset):
    def __init__(self, texts_x, texts_y, tokenizer):
        self.texts_x = texts_x
        self.texts_y = texts_y 
        self.tokenizer = tokenizer
    
    def __getitem__(self, i):
        
        def _process(s):
            s = self.tokenizer(s,
                         max_length=256,
                         padding='max_length',
                         truncation=True,
                         return_tensors='pt')
            return (s['input_ids'].squeeze().to(device), 
                    s['attention_mask'].squeeze().to(device))
            
        x, y = _process(self.texts_x[i]), _process(self.texts_y[i])
        
        return {
            'x_ids': x[0],
            'x_mask': x[1],
            'y_ids': y[0],
            'y_mask': y[1],
        }
        
    def __len__(self):
        return len(self.texts_x)

def train(config, doDebug:bool = False):

    if type(config) != dict: config=_parseConfig(config)
    assert_configLegality(config)

    translateConfig=config['trainingTranslateLM']

    tokenizer = Tokenizer.from_pretrained(translateConfig['modelIdentifier'])
    model = Model.from_pretrained(translateConfig['modelIdentifier']).to(device)

    metric = evaluate.load('sacrebleu')

    def _make_dataset(datalist):
        def _make_input(org, title, question):
            question = '\n'.join(question)
            return f"<{org}> {title}\n{question}"
        X, Y = list(), list()
        for d in datalist:
            X.append(_make_input(d['organization'], d['title'], d['question']))
            Y.append('\n'.join(d['answer']))
            X.append(_make_input(d['organization'], d['title'], d['gen1']))
            Y.append('\n'.join(d['answer']))
            X.append(_make_input(d['organization'], d['title'], d['gen2']))
            Y.append('\n'.join(d['answer']))
            X.append(_make_input(d['organization'], d['title'], d['gen3']))
            Y.append('\n'.join(d['answer']))
        return InqueryDataset(X, Y, tokenizer)

    with open(translateConfig['dataPath']) as f:
        datasets = json.load(f)

    if doDebug:
        _sample = lambda l: random.sample(l, 2)
        datasets['data']['train'] = _sample(datasets['data']['train'])
        datasets['data']['dev'] = _sample(datasets['data']['dev'])
        datasets['data']['test'] = _sample(datasets['data']['test'])

    dataset_train = _make_dataset(datasets['data']['train'])
    dataset_dev = _make_dataset(datasets['data']['dev'])
    dataset_test = _make_dataset(datasets['data']['test'])

    loader_train, loader_dev, loader_test = \
                                DataLoader(dataset_train,
                                          shuffle=True,
                                          batch_size=translateConfig['trainBatch']), \
                                DataLoader(dataset_dev,
                                          shuffle=False,
                                          batch_size=translateConfig['evalBatch']), \
                                DataLoader(dataset_test,
                                          shuffle=False,
                                          batch_size=translateConfig['evalBatch'])

    def _train(loader, 
              optimizer,
              print_step: int=100):
        
        if doDebug: print_step=1
        
        model.train()
        for step, data in enumerate(tqdm(loader,
                                         position=1,
                                         desc="training...")):
            y = data['y_ids'][:, :-1].contiguous()
            lm_labels = data['y_ids'][:, 1:].clone().detach()
            lm_labels[data['y_ids'][:, 1:] == tokenizer.pad_token_id] = -100
            
            ids = data['x_ids'].to(device)
            mask = data['x_mask'].to(device)

            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y, labels=lm_labels)
            loss = outputs[0]
            if not doDebug and translateConfig['reportToWandb']:
                wandb.log({'train_loss':  loss})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step%print_step == 0:
                print(f'Step: {step}, Loss:  {loss.item():04.4f}')
                
    def _validate(loader,
                 print_step: int=100):
        
        if doDebug: print_step=1

        model.eval()
        y_hats = []
        ys = []
        with torch.no_grad():
            for step, data in enumerate(tqdm(loader, 
                                             position=1,
                                             desc="validating...")):
                y = data['y_ids'].to(device)
                y_l = data['y_ids'][:, :-1].contiguous()
                ids = data['x_ids'].to(device)
                mask = data['x_mask'].to(device)
                lm_labels = data['y_ids'][:, 1:].clone().detach()
                lm_labels[data['y_ids'][:, 1:] == tokenizer.pad_token_id] = -100

                generated_ids = model.generate(
                    input_ids = ids,
                    attention_mask = mask,
                )
                loss = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_l, labels=lm_labels)[0]
                
                y_hat = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                y = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
                bleu = metric.compute(predictions=y_hat, references=y, tokenize='char')['score']

                if not doDebug and translateConfig['reportToWandb']:
                    wandb.log({'bleu': bleu,
                                'dev_loss': loss,})

                y_hats.extend(y_hat)
                ys.extend(y)
                
                if step%print_step == 0:
                    print(f'Step: {step}')
                    
        return y_hats, ys

    epochs = 2 if doDebug else translateConfig['maxEpoch'] 

    best_epoch = 1
    best_model = model
    best_bleu = 0.0
    
    if not doDebug and translateConfig['reportToWandb']:
        wandb.init(
            project=translateConfig['modelSaveIdentifier'],
            name=f"run_{datetime.strftime(datetime.now(), 'Y%m%d%H%M')}",
            config={            
                "learning_rate": float(translateConfig['learningRate']),
                "dataset": "/inqueries_aug_1_0",
                "max_new_tokens": 120,
                "repetition_penalty": 1.2,
                "penalty_alpha": 0.7,
                "top_k": 8,
            }
        )

    generation_configs = model.generation_config.to_dict() 

    generation_configs["max_new_tokens"]=10 if doDebug else 120
    # reduced `max_new_tokens` ensures faster debugging
    generation_configs["early_stopping"]=True
    generation_configs["do_sample"]=False,
    generation_configs["num_beams"]=1
    generation_configs["num_beam_groups"]=1
    generation_configs["use_cache"]=False
    generation_configs["temperature"]=1.0
    generation_configs["top_k"]=10
    generation_configs["top_p"]=1.0 # If set to float < 1, only the smallest set of
                # most probable tokens with probabilities that add up to top_p or
                # higher are kept for generation.
                # (https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.top_p)
    generation_configs["diversity_penalty"]=0.0
    generation_configs["repetition_penalty"]=1.2
    generation_configs["length_penalty"]=1.0
    generation_configs["penalty_alpha"]= 0.6

    model.generation_config = GenerationConfig.from_dict(generation_configs)

    for epoch in tqdm(range(translateConfig['epochSince'], translateConfig['epochSince'] + epochs)):
        print(f"Running epoch: {epoch}")
        _train(loader=loader_train, optimizer=torch.optim.Adam(
            params = model.parameters(),
            lr=float(translateConfig['learningRate']),), print_step=100)
        y_hats, ys = _validate(loader=loader_valid, print_step=5)
        bleu = metric.compute(predictions=y_hats, references=ys, tokenize='char')['score']
        if bleu >= best_bleu:
            best_model = model
            best_bleu = bleu
            best_epoch = epoch
        if not doDebug and translateConfig['saveToHF']:
            model.push_to_hub(translateConfig['modelSaveIdentifier'], private=True)
            tokenizer.push_to_hub(translateConfig['modelSaveIdentifier'], private=True)
        if not doDebug and translateConfig['saveLocal']:
            model.push_to_hub(translateConfig['modelSavePath'], private=True)
            tokenizer.push_to_hub(translateConfig['modelSavePath'], private=True)

    if not doDebug and translateConfig['reportToWandb']:
        wandb.finish()

    print(f"Best score: {best_bleu}")
    print(f"Constructing testset with best model at epoch {best_epoch}")

    result = [('X', 'Y_hat', 'Y', 'bleu')]
    for data in tqdm(dataset_test, 
                     position=0,
                     desc="testing..."):
        X = data['x_ids'].tolist()
        X_str = tokenizer.decode(data['x_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        X_mask = data['x_mask'].tolist()
        Y = tokenizer.decode(data['y_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        generated_ids = best_model.generate(
                    input_ids = torch.tensor([X]).to(device),
                    attention_mask = torch.tensor([X_mask]).to(device),
        ).squeeze(0)

        Y_hat = tokenizer.decode(generated_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True)   
        bleu = metric.compute(predictions=[Y_hat], references=[[Y]], tokenize='char')['score']
        result.append((X_str, Y_hat, Y, bleu))
    
    with open('test_result.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(result)

    print("Test result written as test_result.tsv")                
    
    if not doDebug and translateConfig['saveToHF']:
        best_model.push_to_hub(translateConfig['modelSaveIdentifier'] + '-best', private=True)
        tokenizer.push_to_hub(translateConfig['modelSaveIdentifier'] + '-best', private=True)
        
    print(f"Best model saved as {translateConfig['modelSaveIdentifier']}-best")
