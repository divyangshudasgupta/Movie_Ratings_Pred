import streamlit as st
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from PIL import Image

def main():
	st.set_page_config(page_title="Movie Rating Predictor App", layout="centered", page_icon="🎬")
	st.markdown(
	"""
	<style>
	.stApp {
		background-color: #FFCDD2;	/* Light blue color */
	}
	.stButton>button {
		background-color: #FF5733;	/* Light red */
		color: white;  /* White text */
		border: none;  /* Remove default border */
		padding: 10px 20px;	 /* Add padding */
		border-radius: 8px;	 /* Rounded corners */
		font-size: 16px;  /* Font size */
		cursor: pointer;  /* Pointer cursor */
		transition: background-color 0.3s ease;	 /* Smooth hover effect */
	</style>
	""",
	unsafe_allow_html=True)
	
	image = Image.open("cinema.jpg")
	st.image(image, caption="Movies", use_column_width=True)
	st.title("🎬 Movie Rating Predictor App")
	st.write("Predict movie ratings based on genre details,number of votes, and release year.")
	
	# Inputs
	release_year = st.number_input("Please enter the Release year:", min_value=1900, max_value=2025)
	num_votes = st.number_input("Please enter the Number of Votes", min_value=0)
	num_genres = st.number_input("Please enter the number of genres:",min_value=1)
	df = pd.read_csv("data_new.csv")
	mlb = MultiLabelBinarizer()
	df['genres'] = df['genres'].str.strip().str.title()
	df['genres'] = df['genres'].apply(lambda x: x.split(','))
	df['genres'] = df['genres'].apply(lambda genres: [genre.strip() for genre in genres if genre])
	genres_encoded = mlb.fit_transform(df['genres'])
	genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
	
	# Genre selection
	selected_genres = st.multiselect("Enter the genre names", mlb.classes_)
	genres_input = pd.Series(0, index=mlb.classes_)
	genres_input[selected_genres] = 1
	
	# Preparing data for prediction
	input_data = pd.DataFrame([[num_votes,release_year] + list(genres_input)+[num_genres]],
							  columns=['numVotes','releaseYear',] + list(mlb.classes_)+['Num_Genres'])
	
	with open('scaler.pkl', 'rb') as f:
		scaler = pickle.load(f)

	numerical_features = ['releaseYear','numVotes','Num_Genres']
	input_data[numerical_features] = scaler.transform(input_data[numerical_features])
	
	# Loading the saved model
	with open('best_model.pkl', 'rb') as file:
		loaded_model = pickle.load(file)

	if st.button("Predict Rating"):
		if not selected_genres:
			st.warning("Please select at least one genre.")
		else:
			st.success(f"Prediction process initiated")
			
			prediction = loaded_model.predict(input_data)[0]
			st.write(f"Predicted Rating: {prediction:.2f}")
		
	st.markdown("---")
	st.write("Developed with ❤️ by Divyangshu")
		
main()