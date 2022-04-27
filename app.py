from flask import Flask, request, jsonify, render_template
import pickle5 as pickle
import pandas as pd 
import speech_recognition as sr
import re
import sklearn  
import requests
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer 
import io
import os
import requests

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	content = request.json
	remote_url = content['link']
	data = requests.get(remote_url)
	audio=data.content
	audiofile = AudioSegment.from_file(io.BytesIO(audio), format="mp4")
	audiofile.export(out_f="file.wav",format="wav")

	#transcription module begins
	
	recognizer = sr.Recognizer()
	recognizer.energy_threshold = 300
	sample_file = sr.AudioFile('file.wav') 
	#Convert AudioFile to AudioData
	with sample_file as source:
		#record the audio
		sample_audio = recognizer.record(source)
		
	#Transcribing urduSpeech_audio
	trnascripted_text=recognizer.recognize_google(audio_data=sample_audio, language="en-US", show_all= True)
	if type (trnascripted_text) is dict:
		trnascripted_text = trnascripted_text['alternative'][0]['transcript'] 
		trnascripted_text_list=[trnascripted_text]
		sent_transcription = trnascripted_text_list

		app_transcription=pd.DataFrame(trnascripted_text_list, columns=['Transcription'])
		#text based preprocessing
		app_transcription['Transcription'] = app_transcription['Transcription'].apply(lambda x: " ".join(x.lower() for x in x.split())) 
		app_transcription['Transcription'] = app_transcription['Transcription'].map(lambda x: re.sub(r'\W+', ' ', x)) 
		preprocessed_testing_data = app_transcription

		#loading tfidf vetorizer 
		Tfidf_vectorizer = pickle.load(open('./models/vectorizer_word_unigram.pkl', 'rb'))
		# Input of Testing Data
		test_text = preprocessed_testing_data['Transcription']

		# Transform the Input Text using TFIDF Vectorizer
		transform_features = Tfidf_vectorizer.transform(test_text)

		# Get the name of Features (Feature  Set)
		feature_set = Tfidf_vectorizer.get_feature_names_out()

		# Convert Transformed features into Array and Create a Dataframe
		Input_features = pd.DataFrame(transform_features.toarray(), columns = feature_set)

		# Get the name of Features (Feature  Set) and create a DataFrame of Input Features
		input_testing_features = pd.DataFrame(Input_features, columns = Tfidf_vectorizer.get_feature_names_out())
		# Load the Saved Model
		model = pickle.load(open('./models/trained_model.pkl', 'rb'))
		model_predictions = model.predict(input_testing_features)

	else:
		model_predictions = [0]
		sent_transcription = []

	os.remove("file.wav")
	return jsonify( 
		result = int(model_predictions[0]),
		transcription = sent_transcription
	)
