"""construct_data_test.py 

A unit test file supporting `construct_data.py 

Author: Gyu-min Lee 
his.nigel at gmail dot com 
"""

import unittest 
import re

from utils.data import construct_data

class Testconstruct_data(unittest.TestCase):
    
    def test_load_tsv(self):
        content = construct_data._load_tsv('./tests/testInForConstructData.tsv')
        self.assertTrue(all(isinstance(e, construct_data.QAElement) for e in content))
        self.assertTrue(content[0].isHTML == False)
        self.assertEqual(len(content), 3)

    def test_clean_html(self):
        data = construct_data._load_tsv('./tests/testInForConstructData.tsv')
        data = construct_data._clean_html(data)
        self.assertTrue(all(isinstance(e, construct_data.QAElement) for e in data))
        self.assertTrue(all(not d.isHTML for d in data))
        self.assertTrue('<' in data[0].answer)
        self.assertFalse('<' in data[1].answer)

    def test_unescape_html(self):
        data = construct_data._load_tsv('./tests/testInForConstructData.tsv')
        data = construct_data._clean_html(data)
        data = construct_data._unescape_html(data)
        self.assertTrue(all(isinstance(e, construct_data.QAElement) for e in data))
        self.assertTrue(all('&' not in e.answer for e in data))
    
    def test_preprocess(self):
        data = construct_data._load_tsv('./tests/testInForConstructData.tsv')
        data = construct_data._preprocess(data)
        self.assertTrue(all(isinstance(e, construct_data.QAElement) for e in data))
        self.assertTrue(all('☎' not in e.answer for e in data))
        self.assertTrue(all(re.search(r'\[\s+\]', '\n'.join(e.answer)) == None for e in data))

    def test_get_token_counts(self):
        data = construct_data._load_tsv('./tests/testInForConstructData.tsv')
        data = construct_data._clean_html(data)
        data = construct_data._unescape_html(data)
        data = construct_data._get_token_counts(data)
        self.assertTrue(all(isinstance(e, construct_data.QAElement) for e in data))
        self.assertTrue(all(True if e.tokenCount_BART > 0 else False for e in data))
        self.assertTrue(all(True if e.tokenCount_polyglot > 0 or e.answer == "" else False for e in data))
        
    def test_format(self):
        data = construct_data._load_tsv('./tests/testInForConstructData.tsv')
        data = construct_data._preprocess(data)
        data = construct_data._format(data)
        
        self.assertTrue(type(data) == list)
        self.assertTrue(all(type(e) == dict) for e in data)
        
        def _typecheck(d):
            if type(d['qano']) == int and \
                    type(d['organization']) == str and \
                    type(d['title']) == str and \
                    type(d['question']) == list and \
                    type(d['answer']) == list: 
                return True 
            else:
                return False
        self.assertTrue(all(_typecheck(d) for d in data))
        
    def test_split(self):
        data = construct_data._load_tsv('./tests/testInForConstructData.tsv')
        data = construct_data._preprocess(data)
        data = construct_data._format(data)
        data = construct_data._split(data, (1, 1, 1))

        self.assertTrue(len(data[0]) == len(data[1]) == len(data[2]) == 1)
        
if __name__ == '__main__':
    unittest.main()
