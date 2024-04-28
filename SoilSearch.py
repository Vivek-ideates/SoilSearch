import streamlit as st
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as ggi

# to interact with the language model
predefined_responses = {
    "what is SoilSearch?": "SoilSearch provides customized recommendations for optimal crop selection,\n analyzing key soil attributes such as pH, temperature, and moisture levels. \n The data is obtained from Kaggle by ATHARVA INGLE.",
    "How does SoilSearch work?": "SoilSearch works by analyzing key soil attributes such as Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, and rainfall to provide recommendations for optimal crop selection.",
    "Where does SoilSearch get its data from?": "The data for SoilSearch is obtained from Kaggle by ATHARVA INGLE.",
    "Why shouldn't I grow a crop other than the suggested crop?": "You should avoid growing crops that are not well-suited for the current season and your soil conditions because they are unlikely to thrive and produce a good yield. Here are a few key reasons: \n 1) Increased maintenance: Trying to grow ill-suited crops often requires extra inputs like irrigation, fertilizers, pest control, etc. to compensate for the non-ideal conditions, increasing costs and labor. \n 2) Nutrient deficiencies: Soil pH, texture, and nutrient levels may not match the crop's needs when grown out of season, leading to nutrient deficiencies that impact growth and yields. \n 3) Poor germination and stunted growth: Crops have specific temperature, moisture, and soil nutrient requirements for proper germination and growth. Planting outside of their ideal conditions can lead to poor seed emergence, stunted plants, and reduced yields. \n \n Here are a few sources that I found related to this:\n [Source](https://kisanvedika.bighaat.com/news-updates/11-major-problems-faced-by-indian-farmers-in-agriculture-in-2023/)" 
    # Add more predefined questions and answers as needed
}

model = ggi.GenerativeModel("gemini-pro") 
ggi.configure(api_key ="AIzaSyA4Hfgbvycnsh3jOTxvsZAHZo4xBkmmzvg")
chat = model.start_chat()

# Function to interact with the language model
def LLM_Response(question):
    if question in predefined_responses:
        return predefined_responses[question]
    else:
        response = chat.send_message(question, stream=True)
        return response

#CSS for title
css = """
		<style>
			.highlight {
				color: #66ff00;
				animation: fill 5s linear forwards;
				white-space: nowrap;
				overflow: hidden;
				border-right: 1px solid; /* add vertical line */
			}

			@keyframes fill {
				from { width: 0; }
				to { width: 100%; }
			}
			.icon {
            vertical-align: middle;
            padding-right: 10px;
        }
        .card {
			background-color: white;
			border: 5px solid green;
			border-radius: 10px;
			padding: 5px;
			margin: 5px;
			box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
			transition: box-shadow 0.3s ease;
			width:200px;
  		}

		.card:hover {
			box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
		}

		.content {
			text-align: left;
			color: black;
		}

		.label {
			font-weight: bold;
			color: black;
		}
		</style>
		"""

# Display the CSS
st.markdown(css, unsafe_allow_html=True)

# Display the title with streaming effect
st.markdown('<h1 class="highlight">SoilSearch🍀</h1>', unsafe_allow_html=True)

#Declaration of tabs
tab1, tab2, tab3, tab4 = st.tabs(["Crop Recommendation", "Explore other crops", "Community", "About Us"])

# Main app
def main():
	
	with tab1:		
		#sidebar
		st.sidebar.header('User Input Features')
		st.sidebar.markdown("""[Example CSV input file](https://raw.githubusercontent.com/Vivek-ideates/csv-files/main/soil_csv_attributes.csv)""")
		
        #file uploader
		uploaded_file = st.sidebar.file_uploader("Upload your CSV file here!", type=["csv"])
		
  		#read from uploader or from the sliders
		if uploaded_file is not None:
				df = pd.read_csv(uploaded_file)
		else:
			# st.sidebar.markdown("- - -")
			st.sidebar.markdown("<h3 style='text-align: center;'>OR</h3>", unsafe_allow_html=True)
			# st.sidebar.markdown("- - -")
			st.sidebar.write("Manually change the values using the sliders!")
			def user_input_features():
				Nitrogen = st.sidebar.slider('N', 0, 150, 0) 
				phosphorus = st.sidebar.slider('P', 0, 150, 0)
				K = st.sidebar.slider('K', 0, 230, 0)
				temperature = st.sidebar.slider('temperature', 0, 60, 0) 
				humidity = st.sidebar.slider('humidity', 0, 150, 0)
				pH = st.sidebar.slider('ph', 0, 14, 0)
				rainfall = st.sidebar.slider('rainfall', 0, 300, 0)
				data = {'N': Nitrogen,
					'P': phosphorus,
					'K': K,
					'temperature': temperature,
					'humidity': humidity,
					'ph': pH,
					'rainfall': rainfall}
				features = pd.DataFrame(data, index=[0])
				return features

			df = user_input_features()

		if uploaded_file is not None:
			st.write(df)
		else:
			# print("cominghere 3")
			st.write('Awaiting CSV file to be uploaded. Any change to the sliders are reflected below')
			st.write(df)
		crops = pd.read_csv('Crop_recommendation.csv')
		X = crops.drop(columns=['label'])  # Features
		Y = crops['label']  # Target variable

		# Train a RandomForestClassifier
		clf = RandomForestClassifier()
		clf.fit(X, Y)

		prediction = clf.predict(df)
		prediction_proba = clf.predict_proba(df)

		st.subheader('Prediction')
		st.write(prediction)
		st.subheader('Prediction Probability')
		probability_df = pd.DataFrame(prediction_proba, columns=clf.classes_)
		st.write(probability_df)

		# Chat interface
		st.subheader("ChatBot")
		chat_expander = st.expander("Ask Me Questions (click here)", expanded=False)

		with chat_expander:
			# st.title("Ask Me Questions")
			user_quest = st.text_input("Type here:")
			btn = st.button("Submit")

			if btn and user_quest:
				result = LLM_Response(user_quest)
				st.subheader("Response : ")
					# for word in result:
				st.text(result)

if __name__ == "__main__":
    main()


with tab2:
	st.title("Explore Crops")
	st.write("Discover the Maximum and Minimum Range for the Features to Cultivate Your Favorite Crop!")
	cropData = {'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
            'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
            'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
            'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'}

	css_style = """
	<style>
    .popup {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: white;
        border: 1px solid #ccc;
        padding: 20px;
        z-index: 9999;
        box-shadow: 0px 0px 15px 0px rgba(0,0,0,0.5);
    }
</style>
"""

	columns = st.columns(5)
	for i, crop in enumerate(cropData):
		with columns[i % 5]:
			if st.button(crop):
				if crop in cropData:
					file = pd.read_csv('Crop_recommendation.csv')
					crop_data = file[file['label'] == f"{crop}"]
					range_data = crop_data.describe().loc[['min', 'max']].T
					range_data.columns = ['Min', 'Max']

					# Display the range of the label "rice" data
					st.write(f"The range for {crop} is:")
					st.write(range_data)
				else:
					st.write(f"Sorry, the range is not available for {crop}.")


with tab3:
    st.title("Community")
    def card(name, experience, location, distance):
        html = f"""
        <div class="card">
            <div class="content">
                <h2><span class="label">{name}</span></h2>
                <p><span class="label">Experience:</span> {experience} years</p>
                <p><span class="label">Location:</span> {location}</p>
                <p><span class="label">Distance:</span> {distance} km</p>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    st.write("Connect with Nearby Soil Scientists to discover the properties in your soil. Gain insights into crop health, and cultivate success with expert reports and analysis.")
    card("Rahul", 5, "Hyderabad", 13)
    card("Vijay", 1, "Online", 0)
    card("Mahesh", 2, "Delhi", 200)
    card("Arjun", 12, "Secunderabad", 25)

	
with tab4:
    st.title("About Us")
    st.write("""
SoilSearch offers tailored recommendations for optimal crop selection, leveraging advanced analysis of crucial soil attributes such as pH, temperature, and moisture levels.

In addition, SoilSearch features a personalized chat agent, empowering users to inquire about soil properties and receive expert guidance. Explore our vibrant community page to connect with nearby Soil Scientists.

Grounded in the vision of founder Vivek Reddy, SoilSearch emerged as an ambitious endeavor to create an all-encompassing application tailored to meet every user's agricultural soil needs. 

The dataset powering SoilSearch is sourced from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/data) by ATHARVA INGLE.
""")



    






























