import requests
import os
os.chdir(r"C:\Users\wan397\PycharmProjects\Boptest\project1-boptest-gym")
cwd = os.getcwd()
print("Current working directory is:", cwd)
from boptestGymEnv import BoptestGymEnv

# url for the BOPTEST service
url = 'https://api.boptest.net'

# Select test case and get identifier
testcase = 'bestest_hydronic_heat_pump'

# Check if already started a test case and stop it if so before starting another
try:
  requests.put('{0}/stop/{1}'.format(url, testid))

  print('Terminalizing the current test')
except:
  print('No test is processing')
  pass

# Select and start a new test case
testid = \
requests.post('{0}/testcases/{1}/select'.format(url,testcase)).json()['testid']

# Get test case name
name = requests.get('{0}/name/{1}'.format(url, testid)).json()['payload']
print(name)

# Get inputs available
inputs = requests.get('{0}/inputs/{1}'.format(url, testid)).json()['payload']
print('TEST CASE INPUTS ---------------------------------------------')
print(inputs.keys())
# Get measurements available
print('TEST CASE MEASUREMENTS ---------------------------------------')
measurements = requests.get('{0}/measurements/{1}'.format(url, testid)).json()['payload']
print(measurements.keys())
requests.put('{0}/stop/{1}'.format(url, testid))
