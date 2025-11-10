import unittest
import sys
import os

# Add the src directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'src')))

from src.manager.accountManager import AccountManager
from src.models.account import Account

class TestOwnerMethods(unittest.TestCase):
    #setup
    manager = AccountManager()
    a = Account("gmail", "hpeterson@gmail.com")
    
    
    
    
    
if __name__ == '__main__':
    unittest.main()