import unittest
import sys
sys.path.append('../')
from app import app

class FlaskTestCase(unittest.TestCase):

    # Ensure that Flask was set up correctly
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type='html/text')
        self.assertEqual(response.status_code, 200)


    # Ensure the prediction functionality works
    def test_prediction(self):
        tester = app.test_client(self)
        response = tester.post(
            '/process',
            data=dict(weight='70', height='1.75', sight='1', 
                      doctim1y='2', dentim1y='1', oopden1y='0', 
                      oopdoc1y='0', oopmd1y='0', decsib='1', momage='80',
                      cholst='1', fallnum='1', hltc='1'
                      ),
            follow_redirects=True
        )
        # Here, you're just checking for a 200 response.
        # In a real-world scenario, you might want to verify the prediction value as well.
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
