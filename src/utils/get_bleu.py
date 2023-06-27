"""getBleu.py 

Get BLEU scores with character-based sacrebleu from strings.

Author: Gyu-min Lee 
his.nigel at gmail dot com 
"""

import sys
import os
import csv 

from typing import Collection
from tqdm.contrib.concurrent import process_map

import evaluate

def _get_score(preds: Collection, references: Collection,
                doNormalize: bool=False) -> list:
    
    assert isinstance(preds, Collection)
    assert isinstance(references, Collection)
    
    if len(preds) != len(references):
        raise ValueError("Predictions and references should be in the same length.")

    references = [list(i) if type(i) == str else i for i in references]
    # ensures the references format for HF `evaluate`
    
    metric = evaluate.load('sacrebleu')
    
    result = list()

    if doNormalize:
        result.append(metric.compute(predictions=preds,
                       references=references,
                       tokenize='char')['score'])
    else:
        result = [metric.compute(predictions=[pred], 
                                 references=[refer], 
                                 tokenize='char')['score'] 
                  for pred, refer in zip(preds, references)]
            
    return result

def _score_file_lines(args):
    line = args['line']
    predColName = args['predColName']
    referenceColName = args['referenceColName']
    line['score'] = _get_score(preds=[line[predColName]], 
                           references=[line[referenceColName]])[0]
    return line

def score_file(filename,
               predColName, referenceColName,
               outname: str=""):
    
    srcLines = list()
    with open(filename) as f:
        if filename.endswith('tsv'): 
            reader = csv.DictReader(f, delimiter='\t')
            srcLines = [l for l in reader]
        else:
            raise NotImplementedError
    
    srcLines = process_map(_score_file_lines, [{
        "line": line,
        "predColName": predColName,
        "referenceColName": referenceColName,
        } for line in srcLines]) 
    
    outname = outname if outname != "" else '_scored'.join(os.path.splitext(filename))
    
    with open(outname, 'w') as f:
        if outname.endswith('tsv'):
            writer = csv.DictWriter(f, fieldnames=list(srcLines[0].keys()),
                                    delimiter='\t')
            writer.writeheader()
            writer.writerows(srcLines)
        
    return

def _test():
    testPairs = [
            ("대한민국은 한국이라고도 불린다.", "대한민국은 한국이라고도 불린다."), #  expected to be 100.0
            ("대한민국은 한국이라고도 불린다.", "한국은 대한민국이라고도 불린다."), # not 100.0 but close 
            ("대한민국은 한국이라고도 불린다.", "미국은 미합중국이라고 한다."), # expected to be moderate 
            ("대한민국은 한국이라고도 불린다.", "호주는 태평양."), # expected to be near 0.0 
            ]
    scores = _get_score(preds=[i[0] for i in testPairs],
                        references=[[i[1]] for i in testPairs],
                        doNormalize=False)

    for pair, score in zip(testPairs, scores):
        print(pair[0])
        print(pair[1])
        print(score)
        print()
    
    score_normalized = _get_score(preds=[i[0] for i in testPairs],
                        references=[[i[1]] for i in testPairs],
                        doNormalize=True)

    print(score_normalized)

    return

def _run():
    score_file(sys.argv[1], predColName='Y_hat', referenceColName='Y')

if __name__ == "__main__":
    _run()
