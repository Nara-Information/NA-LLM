"""make_openapi_call.py 

Definition of helper functions to make API calls and handle the result.

Author: Gyu-min Lee 
<his.nigel at gmail dot com>
"""

import sys 

import html
import json 

from collections import namedtuple
from typing import NamedTuple
from pprint import pprint

import requests 

from lxml import etree 

_URL_LIST = "http://apis.data.go.kr/1140100/CivilPolicyQnaService/PolicyQnaList"
_URL_INDIV = "http://apis.data.go.kr/1140100/CivilPolicyQnaService/PolicyQnaItem"

IndivResult = namedtuple('IndivResult', 
                         ('qano',
                          'title', 
                          'question', 
                          'answer',
                          'organization',
                          'subject',
                          ))

def _call(url, params):
    response = requests.get(url, params=params)
    try:
        return json.loads(response.content)
    except:
        return response.content

def _stringifyResponse(resp: str) -> str:
    try:
        resp = html.unescape(resp)
        doc = "<doc>\n" + resp + "</doc>"
        doc = doc.replace('&', '&amp;')
        doc = etree.fromstring(doc)
    except etree.XMLSyntaxError:
        return resp
    return ''.join(doc.itertext())

def getList(**kwds) -> list:
    REQUIRED_KEYS = 'serviceKey,firstIndex,recordCountPerPage'.split(',')
    for key in REQUIRED_KEYS:
        assert key in kwds 
    
    result = _call(_URL_LIST, kwds) 
    assert type(result) == dict and 'resultList' in result, \
            f"Error on the server response:\n\t{str(result)}"
    return [d['faqNo'] for d in result['resultList']]

def getIndiv(**kwds) -> NamedTuple:
    REQUIRED_KEYS = 'serviceKey,faqNo,dutySctnNm'.split(',')
    for key in REQUIRED_KEYS:
        assert key in kwds, f"Required key {key} not found"
    
    result = _call(_URL_INDIV, kwds)
    assert type(result) == dict and  'resultData' in result, \
            f"Error on the server response:\n\t{str(result)}"
    result = result['resultData']
    assert (key in result for key in 
                 ('qnaTitl, qstnCntnCl', 'ansCntnCl')
            ), \
            f"Missing required keys from the response:\n\t{str(result)}"
    
    qano = kwds['faqNo']
    title = result['qnaTitl']
    q = result['qstnCntnCl']
    a = _stringifyResponse(result['ansCntnCl'])
    org = result['ancName'] if 'ancName' in result else 'NA'
    subject = result['subjectName'] if 'subjectName' in result else 'NA'

    return IndivResult(qano, title, q, a, org, subject)

def main():
    result = getIndiv(faqNo=sys.argv[1])
    pprint(result)
    return

if __name__ == "__main__":
    main()
