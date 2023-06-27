"""augment_test.py 

A unit test file supporting `train/translate.py`

Author: Gyu-min Lee 
his.nigel at gmail dot com 
"""

import unittest
import traceback

from train import translate

class TestTrain(unittest.TestCase):
    config = translate._parseConfig('config.yaml')
    
    def test_parseConfig(self):
        self.assertEqual(type(self.config), dict)

    def test_train(self):
        try:
            err = None 
            translate.train(self.config, True)
        except BaseException as e:
            err = e
            traceback.print_exc()
        finally:
            self.assertIsNone(err, 
                              "train() had at least one error.")
            
if __name__ == '__main__':
    unittest.main()
