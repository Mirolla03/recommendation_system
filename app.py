import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from fuzzywuzzy import process
import mlflow
import mlflow.tensorflow
import mlflow.pyfunc
import zipfile
import os
extract_path = r"B:\GP\models\cosine"

# Assuming the file is named 'cosine_similarity.npy' after extraction
cosine_sim_path = os.path.join(extract_path, 'arr_0.npy')

# Load the similarity matrix
cosine_sim = np.load(cosine_sim_path)


# Load the cosine similarity matrix
cosine_sim = np.load('cosine_similarity.npy')
movies = pd.read_csv('B:\GP\data\movies.csv')  # Load your movies DataFrame


# Load the models
# Load the models
@st.cache_resource
def load_trained_models():
    # Load the generator model for GAN
    generator_model = tf.keras.models.load_model(r'B:\GP\models\gan_generator_model.h5')
    
    # Load the discriminator model (if needed)
    discriminator_model = tf.keras.models.load_model(r'B:\GP\models\gan_discriminator_model.h5')
    
    # Load the SVD model using joblib
    #svd_model = joblib.load(r'B:\GP\models\svdpp.pkl')
    
    #content_model = ...  # Load your content-based model


    return generator_model, discriminator_model#, svd_model

generator, discriminator = load_trained_models()

# Function to generate synthetic profiles using GAN
def generate_synthetic_profile(latent_dim=35):
    noise = np.random.normal(0, 1, (1, latent_dim))
    synthetic_profile = generator.predict(noise)
    return synthetic_profile

# Load the preprocessed movie dataset
df = pd.read_pickle("encoded_genres.pkl")


# Function to find the closest matching movie title
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

# Function to filter movies based on the selected genres
def filter_movies_by_genres(selected_genres):
    # Assuming the 'genres' column contains a list of genres for each movie
    filtered_movies = movies[movies['genres'].apply(lambda x: any(genre in x for genre in selected_genres))]
    return filtered_movies

# Function to get content-based recommendations considering title and genres
def get_content_based_recommendations(title_string, selected_genres, n_recommendations=10):
    # Find the closest match to the movie title
    title = movie_finder(title_string)
    movie_idx = dict(zip(movies['title'], list(movies.index)))
    idx = movie_idx[title]

    # Filter the movies by genres
    filtered_movies = filter_movies_by_genres(selected_genres)
    
    # Filter the cosine similarity matrix to include only filtered movies
    filtered_movie_indices = filtered_movies.index.tolist()
    
    # Get similarity scores only for filtered movies
    sim_scores = [(i, cosine_sim[idx][i]) for i in filtered_movie_indices]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top n recommendations
    sim_scores = sim_scores[1:(n_recommendations + 1)]
    similar_movies = [i[0] for i in sim_scores]

    return movies['title'].iloc[similar_movies]



def get_gan_recommendations(generator_model, movies_df, top_n=5):
    """
    Generate movie recommendations using a GAN generator model.
    """
    # Use the generator model to create a synthetic user profile
    generated_user_profile = generator_model.predict(np.random.randn(1, 100))  # Example latent space input

    # Based on the generated user profile, we sample or filter movies
    # For simplicity, we'll randomly recommend top_n movies here, but in practice,
    # you would compare the profile with movie features.
    recommended_movies = movies_df.sample(n=top_n)['title'].tolist()
    
    return recommended_movies

# Main Streamlit app logic
# Main Streamlit app logic
def main():
    st.title("Movie Recommendation System")

    # Initialize MLflow run
    with mlflow.start_run(run_name="movie_recommendation_experiment"):

        # Step 1: Dynamic Movie Selection
        movie_list = df['title'].unique()
        fav_movie = st.selectbox("What is your favorite movie?", movie_list)

        # Step 2: Dynamic Genre Selection
        genre_columns = [col for col in df.columns if col not in ['title', 'userId','rating','movie_id']]
        genres = st.multiselect("Select your top 3 favorite genres:", genre_columns)

        st.write(f"Favorite Movie: {fav_movie}")
        st.write(f"Selected Genres: {genres}")

        # Log the selected movie and genres to MLflow
        mlflow.log_param("favorite_movie", fav_movie)
        mlflow.log_param("selected_genres", genres)

        # Step 3: Generate Synthetic Profile
        if st.button('Generate Synthetic Profile'):
            profile = generate_synthetic_profile()
            st.write('Generated Synthetic Profile:', profile)

            # Log the synthetic profile to MLflow
            mlflow.log_artifact("Generated Profile", profile)

        # Step 4: Get Personalized Recommendations
        if st.button("Get Personalized Recommendations"):
            if fav_movie and genres:
                st.write(f"Your favorite movie is: {fav_movie}")
                st.write(f"Your favorite genres are: {', '.join(genres)}")

                # Call recommendation functions
                #svd_recommendations = get_svd_recommendations(user_id, svd_model)  # Collaborative filtering
                content_recommendations = get_content_based_recommendations(fav_movie, genres)
                gan_recommendations = get_gan_recommendations(generator_model, movies_df=df)

                # Display results
                # st.subheader("Collaborative Filtering Recommendations")
                # st.write(svd_recommendations)
                st.subheader("Content-Based Filtering Recommendations")
                st.write(content_recommendations)

                st.subheader("GAN-Based Recommendations")
                st.write(gan_recommendations)

                # Log recommendations as artifacts
                mlflow.log_artifact("Content Recommendations", content_recommendations)
                mlflow.log_artifact("GAN Recommendations", gan_recommendations)

            else:
                st.write("Please provide both your favorite movie and genres.")

if __name__ == '__main__':
    main()
