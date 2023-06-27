"""construct_data.py 

Build seed dataset from OpenAPI

Author: Gyu-min Lee 
his.nigel at gmail dot com 
"""

import os
import sys
import csv 
import json 
import re
import html
import random
import warnings

from random import sample
from datetime import datetime
from collections import namedtuple, deque
from typing import Collection, Literal

from tqdm import tqdm
from transformers import AutoTokenizer as Tokenizer

from .make_openapi_call import getList, getIndiv

random.seed(612)

NA = "N/A"
QAElement = namedtuple("QAElement", field_names=(
    "qano",
    "organization",
    "title",
    "question",
    "answer",
    "tokenCount_BART",
    "tokenCount_polyglot",
    "isHTML"
    ), defaults=(
        0,
        NA,
        NA,
        NA,
        NA,
        0,
        0,
        False
    ))

def _load_json(json_path) -> list:
    """json loader. Should make a list of namedtuple of QAElement out of 
    json based on json dictionaries. 
    
    Returns:
        list: `[QAElement]` 
    """
    
    with open(json_path) as f:
        data = json.load(f)['data']
    
    data = [data[d] for d in data]
    data = [j for i in data for j in i]
        
    return [QAElement(qano=l['qano'], 
                      organization=l['organization'],
                      title=l['title'],
                      question='\n'.join(l['question']),
                      answer='\n'.join(l['answer_cleaned']),
                      tokenCount_BART=0,
                      tokenCount_polyglot=0,
                      isHTML=False,
                    ) for l in data]

def _load_tsv(tsv_path) -> list:
    """tsv loader. Should make a list of namedtuple of QAElement out of 
    tsv based on tsv columns 
    
    Returns:
        list: `[QAElement]` 
    """
    
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t', lineterminator='\n')
        lines = list(reader)
        
    return [QAElement(qano=int(l['qano']), 
                      organization=l['organization'],
                      title=l['title'],
                      question=l['question'],
                      answer=l['answer_cleaned'],
                      tokenCount_BART=int(l['tokenCount_BART']) if 'tokenCount_BART' in l.keys() else 0,
                      tokenCount_polyglot=int(l['tokenCount_polyglot']) if 'tokenCount_polyglot' in l.keys() else 0,
                      isHTML=bool(True if l['isHTML'].lower() == 'true' else False) if 'isHTML' in l.keys() else True if l['subject'].lower()=='html' else False 
                    ) for l in lines]

def _preprocess(data: list) -> list:
    """
    preprocess routines to be applied to string variables in the data.
    """
    
    nbspToSpace = lambda s: s.replace('\u00A0', ' ') 
    removeEmptyLines = lambda s: re.sub(r"\n(?:\s*\n)+", '\n', s) 
    removePhoneNumbers = lambda s: re.sub(r"☎[\d\s\-~,]+", '', s) 
    removeEmptyParagraphs = lambda s: re.sub(r"[\(\[]\W*?[\]\)]", '', s)
    
    removes = lambda s: [e for e in removeEmptyParagraphs(
            removePhoneNumbers(
                removeEmptyLines(
                    nbspToSpace(
                            '\n'.join(s)
                        )
                    )
                )
            ).split('\n') if e != ""];
    
    def _process(d):
        d = d._asdict()
        d['question'] = removes([d['question']])
        d['answer'] = removes([d['answer']])
        d = QAElement(**d)
        return d 
    
    return list(map(_process, data))

def _clean_html(data: list) -> list:
    """
    html cleaner. Remove all < > texts if `isHTML`
    """
    _stripXMLTag = lambda s: re.sub(r'<.+?>', '', s)
    def _process(d):
        d = d._asdict()
        if d['isHTML']:
            d['answer'] = _stripXMLTag(d['answer'])
            d['isHTML'] = False 
        d = QAElement(**d)
        return d 

    return list(map(_process, data))

def _unescape_html(data: list) -> list:
    """
    unescape html for all strings
    """
    def _process(d):
        d = d._asdict()
        d['organization'] = html.unescape(d['organization'])
        d['title'] = html.unescape(d['title'])
        d['question'] = html.unescape(d['question'])
        d['answer'] = html.unescape(d['answer'])
        d = QAElement(**d)
        return d 
    
    return list(map(_process, data))

def _get_token_counts(data: list, models: list=[
    "hyunwoongko/kobart",
    "EleutherAI/polyglot-ko-1.3b"]) -> list:
    """
    put token counts (for the answer) with KoBART and polyglot-ko tokenizer
    """
    
    print("Loading tokenizers")
    tokenizers = [Tokenizer.from_pretrained(model) for model in models]
    print(f"{len(tokenizers)} tokenizers loaded")

    def _process(d):
        d = d._asdict()
        d['tokenCount_BART'] = len(tokenizers[0](d['answer']).input_ids)
        d['tokenCount_polyglot'] = len(tokenizers[1](d['answer']).input_ids)
        d = QAElement(**d)
        return d 
    
    return list(map(_process, data)) 

def _format(data: list) -> list:
    """
    format object to be complied with the JSON format. 
    Final JSON object should be:
    ```
        [
            {
                qano: number,
                organization: text,
                title: text,
                question: text,
                answer: text,
            }
        ]
    ```
    """
    
    def _condition(d):
        if d.tokenCount_BART >= 516: return False
        elif re.search(r'\w', ''.join(d.answer)) == None: return False 
        else: return True
    
    result = list()

    for d in data:
        if _condition(d):
            result.append({
                "qano": int(d.qano),
                "organization": d.organization,
                "title": d.title,
                "question": d.question,
                "answer": d.answer,
                })
     
    return result 

def _split(data: list, ratio: tuple=(7, 2, 1)) -> tuple:
    datacount = len(data)
    
    train_size = int(datacount * ratio[0] / sum(ratio))
    dev_size = int(datacount * ratio[1] / sum(ratio)) 
    test_size = datacount - train_size - dev_size
    
    indexes = sample(range(datacount), k=datacount)
    train_ind = indexes[:train_size]
    dev_ind = indexes[train_size:train_size+dev_size]
    test_ind = indexes[train_size+dev_size:] 
    
    train, dev, test = list(), list(), list()
    for i in range(len(data)):
        if i in train_ind:
            train.append(data[i])
        elif i in dev_ind:
            dev.append(data[i])
        elif i in test_ind:
            test.append(data[i])
        else:
            pass 
    
    return sample(train, len(train)), sample(dev, len(dev)), sample(test, len(test))

def _save(data: Collection|dict, outname: str, mode: Literal['tsv', 'json']):
    if mode == 'tsv':
        data = [d._asdict() for d in data]
        with open(outname + '.' + mode, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys(), 
                                    
                                    delimiter='\t')
            writer.writeheader()
            writer.writerows(data)
    elif mode == "json":
        with open(outname + '.' + mode, 'w') as f:
            data = {
                    "dataset": "민원_openapi_200101-230531",
                    "lastupdate": datetime.now().isoformat(),
                    "datacount": {
                        "train": len(data['train']),
                        "dev": len(data['dev']),
                        'test': len(data['test']),
                        'total': len(data['train']) + len(data['dev']) + len(data['test'])
                        },
                    "data": data,
                    }
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    return 

def make(key:str, n_sample:int, daterange:tuple=('','')):
    # @TODO add ignore by qano feature 
    
    assert len(daterange) == 2, \
            "expected two dates for `daterange`"
    datefrom, dateto = daterange
    
    try:
        if datefrom != '' and dateto != '':
            qaNos = getList(serviceKey=key, 
                            firstIndex='1', 
                            recordCountPerPage='1000', 
                            dutySctnNm='tqapttn',
                            regFrom=datefrom,
                            regTo=dateto,
                            )
        else:
            qaNos = getList(serviceKey=key, 
                            firstIndex='1', 
                            recordCountPerPage='1000', 
                            dutySctnNm='tqapttn',
                            )
    except AssertionError as e:
        print("Assertion error occurred as:")
        print(e)
        print("Handle and run again.")
        raise RuntimeError
    
    qaNos = sorted(qaNos, reverse=True)
    qaNos = deque(qaNos)
    
    result = list()
    with tqdm(total=n_sample, desc="Getting data from OpenAPI...") as tqdmbar:
        while len(result) < n_sample:
            qaNo = qaNos.pop()
            try:
                result.append(getIndiv(serviceKey=key,
                                       faqNo=qaNo,
                                       dutySctnNm='tqapttn'))            
                tqdmbar.update(1)
            except AssertionError as e:
                print("Assertion error occurred as:")
                print(e)
                print("Skipping the data")
            if len(qaNos) == 0:
                print("Stop making samples since no more qa is left")
                break
    return result

def clean(data, make_tsv, make_json, outpath, splitratio):
    data_bkup = data

    if make_tsv:
        data = _clean_html(data)
        data = _unescape_html(data)
        data = _get_token_counts(data)
        _save(data, outpath+'.tsv', mode="tsv")
    
    data = data_bkup 
    if make_json:
        data = _preprocess(data)
        data = _format(data)
        train, dev, test = _split(data, ratio=splitratio)
        data = {
                "train": train,
                "dev": dev,
                "test": test,
                }
        _save(data, outpath+'.json', mode="json")
    
    return

def construct_data(args, creds):
    data = list()
    if args['continueFromPreviousFile']:
        if args['previousFilePath'].endswith('.tsv'):
            data = _load_tsv(args['previousFilePath'])
        elif args['previousFilePath'].endswith('.json'):
            data = _load_json(args['previousFilePath'])
        else:
            raise RuntimeError("Previous file must be in tsv or json.")
    if args['doFetch']:
        if data != list():
            warnings.warn("doFetch was set while previous data were loaded. \
                          Be aware that there will be no duplicate check;\
                          fetched data will be appended to the existing data.\
                          Set continueFromPreviousFile as False to avoid this issue.")
        data.extend(make(creds['OpenAPI'], args['nMax'],
                    daterange=(str(args['searchFrom']), str(args['searchTo']))))
    if args['doClean']:
        if data == list():
            raise RuntimeError("Got no data to clean.")
        clean(data, 
              args['makeReviewTsv'], 
              args['makeCompleteJson'], 
              os.path.join(args['outputPath'], args['outputName']),
              splitratio=tuple(args['dataSplitRatio'].values()))
    
    return 