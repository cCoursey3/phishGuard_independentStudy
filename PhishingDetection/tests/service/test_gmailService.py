
import unittest
from unittest.mock import patch, MagicMock
from googleapiclient.errors import HttpError
import sys
import os

# Add the src directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'src')))



class TestGmailService(unittest.TestCase):
    @patch('googleapiclient.discovery.build')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"token": "fake_token", "refresh_token": "fake_refresh_token", "token_uri": "fake_uri", "client_id": "fake_id", "client_secret": "fake_secret"}')
    @patch('os.path.exists')
    @patch('json.load')
    def test_initialization(self, mock_json_load, mock_exists, mock_open, mock_build):
        # Setup mocks
        mock_exists.return_value = True  # Simulate that token.json exists
        fake_token_info = {
            'token': 'fake_token',
            'refresh_token': 'fake_refresh_token',
            'token_uri': 'fake_uri',
            'client_id': 'fake_id',
            'client_secret': 'fake_secret'
        }
        mock_json_load.return_value = fake_token_info

        # Mock the build function to return a service with a stubbed users().messages().list() method
        mock_service = MagicMock()
        mock_users = MagicMock()
        mock_messages = MagicMock()
        mock_list = MagicMock()
        mock_service.users.return_value = mock_users
        mock_users.messages.return_value = mock_messages
        mock_messages.list.return_value = mock_list
        mock_list.execute.return_value = {'messages': [{'id': '123', 'threadId': '456'}]}

        mock_build.return_value = mock_service

        # Initialize the service
        from src.service.gmailService import GmailService
        service = GmailService()
        
        # Verify that build is called and credentials are set up correctly with fake data
        mock_build.assert_called_once()
        self.assertEqual(service.credentials.token, 'fake_token')

if __name__ == '__main__':
    unittest.main()
