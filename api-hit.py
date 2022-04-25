import json
import requests

url = 'http://127.0.0.1:5000/predict'
myobj = {'link': 'https://firebasestorage.googleapis.com/v0/b/virtuousvoice-7efd1.appspot.com/o/toxicData%2F%2B923230000000%2Fgama%2F1650539262875.mp3?alt=media'}
x = requests.post(url, json = myobj)
print(x.text)