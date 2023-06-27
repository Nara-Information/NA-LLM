"""augment_test.py 

A unit test file supporting `augment.py`

Author: Gyu-min Lee 
his.nigel at gmail dot com 
"""

import os
import json
import unittest 

from utils.configs import get_config, set_cred
from utils.data.augment import augment

global KEY

class TestAugment(unittest.TestCase):
    
    def test_augment(self):
        configs = get_config('config.yaml')
        cred = set_cred(configs['credPath'])
        
        augment(configs['augmenting'], cred, doTest=True)
        self.assertTrue('augmented_test.json' in os.listdir(configs['augmenting']['outputPath']))
        with open(os.path.join(configs['augmenting']['outputPath'], 'augmented_test.json')) as f:
            testout = json.load(f) 
        self.assertEqual(type(testout), dict)
        self.assertEqual(len(testout['data']['train']), 3)
        self.assertTrue(all('response' in d for d in testout['data']['train']))

if __name__ == '__main__':
    unittest.main()