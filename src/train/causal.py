"""causal.py
version: 1.1.0

Fine-tuning with CausalLM objective, with an option to use 
QLoRA-based quantized loading for gigantic models.

Reference:
https://colab.research.google.com/drive/12albqRjQO7Th-d60W88G1O9y3ED5TQEh?usp=sharing

Version history
1.0 model is trainable with the script
1.0.1
   - added support for unloading peft model
   - model input now have the answer initials in testing 
1.1.0
    - tokenizer now will add 'end of text' token

Author: Gyu-min Lee
his.nigel at gmail dot com
"""

import sys
import json
import csv
import random

from typing import Collection, Hashable
from datetime import datetime

import yaml 
import torch
import evaluate
import wandb

from tqdm import tqdm
from peft import LoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training

from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

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

    assert 'trainingCausalLM' in config
    causalConfig = config['trainingCausalLM']
    assert 'modelIdentifier' in causalConfig 
    assert 'dataPath' in causalConfig
    assert 'saveLocal' in causalConfig
    if causalConfig['saveLocal']:
        assert 'modelSavePath' in causalConfig, \
            "modelSavePath must be specified if saveLocal is set."
    assert 'saveToHF' in causalConfig
    if causalConfig['saveToHF']:
        assert 'modelSaveIdentifier' in causalConfig, \
            "modelSaveIdentifier must be specified if saveToHF is set."
    assert 'doApplyQuantization' in causalConfig
    if causalConfig['doApplyQuantization']:
        assert 'quantizationBits' in causalConfig, \
            "quantizationBits must be specified if doApplyQuantization is set"
        assert int(causalConfig['quantizationBits']) in [4, 8], \
            "quantizationBits must be 4 or 8 for 4bits and 8bits quantization."
        assert 'doMergeAndUnload' in causalConfig,\
            "doMergeAndUnload must be if quantization is applied."
    assert 'maxStep' in causalConfig
    assert 'trainBatch' in causalConfig 
    assert type(causalConfig['trainBatch']) == int, \
        "batch size should be an integer."
    if 'evalBatch' in causalConfig:
        assert type(causalConfig['evalBatch']) == int, \
            "batch size should be an integer."
    assert 'evalWith' in causalConfig
    evalWith = causalConfig['evalWith']
    if type(evalWith) != list:
        evalWith = [evalWith]
    assert False not in [True if e.lower() in ['bleu', 'sts', 'none'] else False for e in evalWith], \
        f"evalWith must be a combination of 'BLEU', 'sts', or 'none' but got {evalWith}"
    if 'sts' in evalWith:
        assert 'stsModelIdentifier' in causalConfig
    if 'bleu' in evalWith:
        assert 'bleuMethod' in causalConfig
        assert causalConfig['bleuMethod'] in ['sacrebleu', 'custom'], \
            "bleuMethod must be 'sacrebleu' or custom"
        if causalConfig['bleuMethod'] == 'custom':
            assert 'bleuScript' in causalConfig, \
                "The script file defining the bleu score should be specified."
    assert 'reportToWandb' in causalConfig
    if causalConfig['reportToWandb']:
        assert 'wandbProjectName' in causalConfig
        assert 'wandbRunName' in causalConfig 
        if 'wandbConfig' in causalConfig:
            assert type(causalConfig['wandbConfig']) == dict, \
                    f"'wandbConfig' expected to be of type dict but got {type(causalConfig['wandbConfig'])}"
    return

def _loadModels(config) -> tuple: 
    """
    Loader for the model and the tokenizer.
    
    Returns:
        tuple. Tuple of length 2, which consist of: 
            (model, tokenizer)
    """
    causalConfig = config['trainingCausalLM']
    tokenizer = AutoTokenizer.from_pretrained(causalConfig['modelIdentifier'],
    )
    
    if causalConfig['doApplyQuantization']:
        model = AutoModelForCausalLM.from_pretrained(
                causalConfig['modelIdentifier'],
                use_cache=False if causalConfig['doApplyQuantization'] else True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True if causalConfig['quantizationBits'] == 4 else False,
                    load_in_8bit=True if causalConfig['quantizationBits'] == 8 else False,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    ),
                device_map='auto'
                )
        generation_configs = model.generation_config.to_dict() 
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=causalConfig['quantizationBits'],
            lora_alpha=32,
            target_modules=["query_key_value"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        ))
    else:
        model = AutoModelForCausalLM.from_pretrained(causalConfig['modelIdentifier'],
                                                     device_map="auto")
        generation_configs = model.generation_config.to_dict() 

    generation_configs["max_new_tokens"]=128
    generation_configs["early_stopping"]=True
    generation_configs["do_sample"]=False
    generation_configs["num_beams"]=2
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

    model.generation_config = GenerationConfig.from_dict(generation_configs)
    return (model, tokenizer)
    
def _loadData(config) -> dict: 
    """
    Loader for the data.
    Data should be a JSON constructed as:
        {
            ...
            data: {
                split_name: [ ... entry with 
                    'text' field or text literal ... ]
            }
            ...
        }

    Returns:
        dict. Dictionary of datasets where key is split name. If no split, 
              all data are included under the key `all`
    """
    
    delimiter = '\n----답변----\n'    
    with open(config['trainingCausalLM']['dataPath']) as f:
        datasets = json.load(f)
    datasets = datasets['data']
    
    _join_text = lambda x: '\n'.join(x)
    _make_Xs = lambda x: (f"<{x['organization']}> {x['title']}\n{_join_text(x['question'])}",
                            f"<{x['organization']}> {x['title']}\n{_join_text(x['gen1'])}",
                            f"<{x['organization']}> {x['title']}\n{_join_text(x['gen2'])}",
                            f"<{x['organization']}> {x['title']}\n{_join_text(x['gen3'])}",)
    _make_X_and_Ys = lambda x: (delimiter.join((i, _join_text(x['answer']))) for i in _make_Xs(x))
    _flatten = lambda x: [j for i in x for j in i]

    trainset = _flatten(list(map(_make_X_and_Ys, datasets['train'])))
    devset = _flatten(list(map(_make_X_and_Ys, datasets['dev'])))
    testset = _flatten(list(map(_make_X_and_Ys, datasets['test'])))

    return {
            'train': trainset,
            'dev': devset,
            'test': testset,
            'delimiter': delimiter,
            }

def _train(model, tokenizer, config, 
           trainsplit: Collection,
           evalsplit: Collection=list(),
           textfield: str=""):
    """
    Train the model.

    Params:
        triansplit, evalsplit (Collection): an collection data split with tokenizable text.
                              All tokenized text in this parameter will be used
                              for training. Split and preprocess before pass.     
                              If evalsplit is empty, will skip evaluation.
        textfield (str): field name for the text element in the data. Will work 
                         only when elements in datasplit are Hashable. If elements
                         are not Hashable, or if `textfield` is an empty string,
                         will try tokenize the entire element, treating it 
                         to be a Iterable of strings.
    Returns:
        nothing. Training will update the model parameters in place while saving 
                 them from time to time.
    """
    causalConfig = config['trainingCausalLM']

    tokenizer.pad_token = tokenizer.eos_token

    trainsplit = list(map(lambda s: tokenizer(s + tokenizer.eos_token, return_token_type_ids=False), [e[textfield] if type(e) != str else e for e in trainsplit]))
    evalsplit = list(map(lambda s: tokenizer(s + tokenizer.eos_token, return_token_type_ids=False), [e[textfield] if type(e) != str else e for e in evalsplit]))
    
    max_steps = causalConfig['maxStep']
    warmup_steps = 50 if max_steps > 500 else int(max_steps / 10)
    logging_steps = int(max_steps/100) if max_steps < 10_000 else int(max_steps/1000)
    if logging_steps == 0: logging_steps = 10

    if causalConfig['reportToWandb']:
        wandb.init(project=causalConfig['wandbProjectName'],
                   name=causalConfig['wandbRunName'],
                   config=causalConfig['wandbConfig'] if 'wandbConfig' in causalConfig else None)

    trainer = Trainer(
            model=model,
            train_dataset = trainsplit,
            eval_dataset = evalsplit,
            args=TrainingArguments(
                per_device_train_batch_size=causalConfig['trainBatch'],
                per_device_eval_batch_size=causalConfig['evalBatch'] if 'evalBatch' in causalConfig else causalConfig['trainBatch'],
                gradient_accumulation_steps=4,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                learning_rate=1e-4,
                fp16=True,
                logging_steps=logging_steps,
                output_dir="hfrun",
                optim="paged_adamw_8bit" if causalConfig['doApplyQuantization'] else 'adamw',
                report_to="wandb" if causalConfig['reportToWandb'] else "none",
                ),
            data_collator=DataCollatorForLanguageModeling(tokenizer, 
                mlm=False)
            ) 
    
    trainer.train()
    
    if causalConfig['reportToWandb']:
        wandb.finish()
     
def _test(model, tokenizer, config,
          testsplit: Collection,
          data_split_delimiter: str,
          textfield: str="") -> list:
    """
    Have the model to generate for the testsplit. 
    
    Params:
        testsplit (Collection): the test split to pass to the model.
        data_split_delimter (str): a string separating the X tokens and Y tokens.
        textfield (str): the textfield name for the tokenizable text element. Will
                         work iff elements in the testsplit are Hashable. If elements
                         in testsplit are not Hashable, or if textfield is an empty string,
                         testsplit elements will be attempted to be tokenized as a whole.
    Returns:
        list of tuple. Each tuple contains input text, answer to the question, and 
                        preset metrics.
        """
    NO_CUDA = True

    causalConfig = config['trainingCausalLM']
    evalWith = causalConfig['evalWith'] 
    if type(evalWith) == str: evalWith = [evalWith]
    evalWith = [e.lower() for e in evalWith]

    if causalConfig['bleuMethod'] == 'sacrebleu':
        metric = evaluate.load('sacrebleu')
    else:
        raise NotImplementedError("Only `sacrebleu` is supported.")

    model.eval()
    model.config.use_cache = True 
    if not NO_CUDA: model.to('cuda:0')

    result = [('X', 'Y_hat', 'Y', 'BLEU', 'sts', )]
    if textfield != "" and all(isinstance(e, Hashable) for e in testsplit):
        testsplit = [e['textfield'] for e in testsplit] 
    
    for instance in tqdm(testsplit, desc="testing..."):
        X, Y = instance.split(data_split_delimiter)
        X += data_split_delimiter
        X_tokens = tokenizer(X, return_tensors='pt', return_token_type_ids=False)
        X_ids = X_tokens['input_ids'] if NO_CUDA else X_tokens['input_ids'].to('cuda:0')
        X_masks = X_tokens['attention_mask'] if NO_CUDA else X_tokens['attention_mask'].to('cuda:0')
        Y_hat = tokenizer.decode(model.generate(input_ids=X_ids, attention_mask=X_masks,
                               )[0])
        Y_hat = Y_hat.split(data_split_delimiter)[1] if data_split_delimiter in Y_hat else Y_hat
        bleu = metric.compute(predictions=[Y_hat], references=[[Y]], tokenize='char')['score'] if 'bleu' in evalWith else 0.0  
        sts = 0.0 
        result.append((X, Y_hat, Y, bleu, sts))
    return result
    
def _save(model, tokenizer, config):
    """
    Save the model in accordance to the `config` contents.
    """

    try:
        model_base = model.merge_and_unload() \
            if config['trainingCausalLM']['doMergeAndUnload'] else None 
    except:
        model_base = None 
    
    if config['trainingCausalLM']['saveLocal']:
        model.save(config['trainingCausalLM']['modelSavePath'])
        tokenizer.save(config['trainingCausalLM']['modelSavePath'])
        if model_base != None:
            model_base.save(config['trainingCausalLM']['modelSavePath'] + '-base')
            tokenizer.save(config['trainingCausalLM']['modelSavePath'] + '-base')
    if config['trainingCausalLM']['saveToHF']:
        model.push_to_hub(config['trainingCausalLM']['modelSaveIdentifier'],
                          tasks='text-generation',
                          private=True)
        tokenizer.push_to_hub(config['trainingCausalLM']['modelSaveIdentifier'],
                          tasks='text-generation',
                          private=True)
        if model_base != None:
            model_base.push_to_hub(config['trainingCausalLM']['modelSaveIdentifier'] + '-base',
                          tasks='text-zgeneration',
                          private=True)
            tokenizer.push_to_hub(config['trainingCausalLM']['modelSaveIdentifier'] + '-base',
                          tasks='text-generation',
                          private=True)
    
    return

def train(config="config.yaml"):
    if type(config) == str:
        config = _parseConfig(config)
    
    assert_configLegality(config)

    model, tokenizer = _loadModels(config)
    dataset = _loadData(config)
    
    _train(model, tokenizer, config, 
           dataset['train'], dataset['dev'])
    
    _save(model, tokenizer, config)

    print("Model saved; now testing")

    testResult = _test(model, tokenizer, config,
          dataset['test'], dataset['delimiter'])
    
    with open(f'test_result_{datetime.strftime(datetime.now(), "%Y%m%d%H%M")}.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(testResult)
    
    print("Test saved.")
    
    return

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Expecting at least one argument for the mode."
    if sys.argv[1] == "test":
        test()
    elif sys.argv[1] == "run":
        train()
    else:
        raise RuntimeError("Supported modes: 'test', 'run'")
