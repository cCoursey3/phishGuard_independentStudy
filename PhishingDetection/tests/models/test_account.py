import unittest
from PhishingDetection.src.models.account import Account, NotSupportedService, EmptyEmailException


"""
This module provides testing functionalities for the account class
Here setters, getters, and class methods are tested.
"""


class TestAccountMethods(unittest.TestCase):

    def setUp(self):
        self.acc = Account("aol", "HSmith@aol.com")
        self.acc2 = Account("gmail", "chloe@gmail.com", 1234, "access", "refresh", "expiry", "uri")

        for attribute in ['access_token', 'expiry', 'token_uri', 'id', 'refresh_token']:
            self.assertIsNotNone(getattr(self.acc2, attribute))
            
    def test_createValidAccount(self):
        #setup with a valid Account
        acc = Account("outlook", "HSmith@outlook.com")
        self.assertEqual("HSmith@outlook.com", acc._emailAddress)
        self.assertFalse(acc.active_account)    
        
        for attribute in ['access_token', 'expiry', 'token_uri', 'id', 'refresh_token']:
            self.assertIsNone(getattr(acc, attribute))
            
    def test_InvalidAccount(self):
        with self.assertRaises(NotSupportedService):
            Account("Verizon", "chloe@verizon.net") 
         
        with self.assertRaises(EmptyEmailException):
            Account("outlook", None)
         
    def test_idSetter(self):
        self.acc.id = 12234
        self.assertEqual(12234, self.acc.id)
    
    def test_expirySetter(self):
        self.acc.expiry = "practice-expiry-token"
        self.assertEqual("practice-expiry-token", self.acc.expiry)
          
    def test_tokenURISetter(self):
        self.acc.token_uri = "practice-token-uri.com"
        self.assertEqual("practice-token-uri.com", self.acc.token_uri)
    
    def test_refresh_access_TokenSetter(self):
        self.acc.refresh_token = "refresh.Test"
        self.acc.access_token = "access.Test"
        self.assertEqual("refresh.Test", self.acc.refresh_token)
        self.assertEqual("access.Test", self.acc.access_token)
    
    def test_serviceSetter(self):
        self.acc.service = "gmail"
        self.assertEqual("gmail", self.acc.service)
    
    def test_activeSetter(self):
        self.acc.active_account = True
        self.assertTrue(self.acc.active_account)
    
    
    def test_str(self):
        self.assertEqual(f"The email service is aol\n The email address is HSmith@aol.com", str(self.acc))
         
         
         
         
         
         
         
if __name__ == '__main__':
    unittest.main()
    
    
   