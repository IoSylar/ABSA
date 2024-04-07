import streamlit as st
from openai import OpenAI
import json
import pandas as pd
import matplotlib.pyplot as plt

# Define your OpenAI API key
api_key = "insert here the api_key"

# Initialize OpenAI client
client = OpenAI(api_key=api_key)


# Function to perform aspect sentiment analysis
def aspect_sentiment_analysis(text):
    prompt = "Do an Aspect based sentiment analysis with tone detection, you must extract entities and the sentiment related to the entities, the sentiment can be positive, negative and neutral. In addition, you have to extract the tone related to the entity for example between: Angry, Joy, Sad, Fear,None. Show also a reason for the sentiment extracted. For example:\n\nThe ice cream is really god  but the pizza taste bad! horrible!. \n\n PLEASE ANSWER IN THE LANGUAGE OF THE INPUT TEXT \n\n Result an array of dict  and  nothing else :\
    [{\"entity\": \"ice cream\", \"sentiment\": \"positive\", \"reason\": \"because is good\",\"tone\":\"None\"},{\"entity\": \"pizza\", \"sentiment\": \"negative\", \"reason\": \"described as bad\",\"tone\":\"Angry\"}] "

    # Request completion from OpenAI
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": str(prompt)},
            {"role": "user", "content": f"Classify the following: {text}"}
        ]
    )

    # Extracting result from completion
    absa_list = completion.choices[0].message.content
    absa_list_json = json.loads(absa_list)
    return absa_list_json


# Streamlit UI
def main():
    st.title("Aspect-Based Sentiment Analysis")

    # Input text area for user to input text
    text_input = st.text_area("Enter your text:")

    # Perform analysis when user clicks the button
    if st.button("Analyze"):
        # Perform aspect sentiment analysis
        results = aspect_sentiment_analysis(text_input)

        # Creazione di una tabella con i risultati
        st.write("## Tabella dei Risultati")
        df = pd.DataFrame(results)

        # Color the sentiment column
        def color_sentiment(sentiment):
            if sentiment == "positive":
                return "color: green"
            elif sentiment == "negative":
                return "color: red"
            else:
                return "color: grey"

        # Apply the color function to the sentiment column
        df_styled = df.style.applymap(lambda x: color_sentiment(x), subset=["sentiment"])

        # Display the styled table
        st.write(df_styled)

    # File uploader for uploading text files
    st.write("## Upload Reviews from a File")
    uploaded_file = st.file_uploader("Choose a file", type=['txt'])

    if uploaded_file is not None:
        # Read the content of the uploaded file
        file_contents = uploaded_file.getvalue().decode("utf-8")
        reviews = file_contents.split('\n')

        # Dictionary to store sentiment counts for each entity
        sentiment_counts = {}

        # Process each review
        for review in reviews:
            if review.strip() != '':
                review_results = aspect_sentiment_analysis(review)
                for result in review_results:
                    entity = result['entity']
                    sentiment = result['sentiment']
                    if entity not in sentiment_counts:
                        sentiment_counts[entity] = {'positive': 0, 'negative': 0, 'neutral': 0}
                    sentiment_counts[entity][sentiment] += 1

        # Create a DataFrame from sentiment_counts dictionary
        df_sentiment_counts = pd.DataFrame.from_dict(sentiment_counts, orient='index')

        # Plot the bar plot
        st.write("## Bar Plot of Sentiment Counts for Entities")
        st.bar_chart(df_sentiment_counts)

if __name__ == "__main__":
    main()
