import unittest
import sys
import os
from PhishingDetection.src.models.owner import Owner, InvalidName, InvalidPIN


class TestOwnerMethods(unittest.TestCase):
    def test_ownerValid(self):
        o = Owner("None", "None", 1234)
        self.assertEqual("None", o._firstName)
        self.assertEqual("None", o._lastName)
        assert o._pin == 1234
        
        o.firstName = "Chloe"
        o.lastName = "Coursey"
        
        self.assertEqual("Chloe", o._firstName)
        self.assertEqual("Coursey", o._lastName)
        
        self.assertEqual("Chloe Coursey", str(o))
        
     
    def test_ownerInvalidFirstNames(self):
        
        with self.assertRaises(ValueError):
            Owner(None, "None", 1234) 
        
        
        o = Owner("None", "None", 1234)
        #try an invalid first name with a number
        with self.assertRaises(InvalidName):
            o.firstName = "1Chloe"
        
        #try an invalid first name with a symbol
        with self.assertRaises(InvalidName):
            o.firstName = "@&Chloe#"
            
            
        with self.assertRaises(InvalidName):
            o.firstName = "3Chl@e"

        
    def test_ownerInvalidLastName(self):
        with self.assertRaises(ValueError):
            Owner("None", None, 1234) 
        
        o = Owner("None", "None", 1234)
        
        with self.assertRaises(InvalidName):
            o.lastName = "1Coursey"
        
        
        with self.assertRaises(InvalidName):
            o.lastName = "@&Smith#"
        
        
        with self.assertRaises(InvalidName):
            o.lastName = "M!ll3r"

    
    def test_ownerInvalidPin(self):
        with self.assertRaises(ValueError):
            Owner("Jessice", "Smith", None)
        
        with self.assertRaises(InvalidPIN):
            Owner("Jessica", "Jackson", 123)
        
        with self.assertRaises(InvalidPIN):
            Owner("Ben", "Tompson", "1%59LkM") 

        o = Owner("Jess", "Frederick", "12345")
        with self.assertRaises(InvalidPIN):
            o.pin = 12
            
        with self.assertRaises(InvalidPIN):
            o.pin = 1222222222222222
            
    def test_string(self):
        o = Owner("Jess", "Frederick", "12345")
        self.assertEqual("Jess Frederick", str(o))
      
    
    
        
        
        
        
        
        

if __name__ == '__main__':
    unittest.main()
