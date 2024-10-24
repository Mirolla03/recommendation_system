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
from sklearn.metrics.pairwise import cosine_similarity
import gc
import psutil

def get_svdpp_recommendations(user_id, model, rating_df, movies_df, top_n=5):
    """
    Recommends top N movies for a specific user using the SVDpp model.

    :param user_id: ID of the user
    :param model: Trained SVDpp model
    :param rating_df: DataFrame containing userId, movieId, and rating
    :param movies_df: DataFrame containing movieId and title
    :param top_n: Number of movies to recommend (default: 10)
    :return: List of top N recommended movie titles
    """
    # Get all movie ids from the movies DataFrame
    all_movie_ids = movies_df['movieId'].unique()  # Ensure 'movieId' is correct

    # Get the list of movies that the user has already rated
    watched_movie_ids = rating_df[rating_df['userId'] == user_id]['movieId'].unique()

    # Find movies the user hasn't watched yet
    not_watched_ids = [mid for mid in all_movie_ids if mid not in watched_movie_ids]

    # Predict ratings for all unwatched movies
    predicted_ratings = [model.predict(user_id, mid) for mid in not_watched_ids]

    # Sort movies by predicted rating in descending order
    predicted_ratings.sort(key=lambda x: x.est, reverse=True)

    # Get the top N movie ids
    top_movie_ids = [pred.iid for pred in predicted_ratings[:top_n]]

    # Return the movie titles for the top N recommendations
    recommended_titles = movies_df[movies_df['movieId'].isin(top_movie_ids)]['title'].tolist()


    return list(set(recommended_titles))



def calculate_similarity_in_chunks(matrix, chunk_size=8000):
    n_samples = matrix.shape[0]
    n_chunks = (n_samples // chunk_size) + 1
    similarities = np.zeros((n_samples, n_samples))

    def get_memory_usage():
        return psutil.Process().memory_info().rss / 1024 / 1024  # MB

    initial_memory = get_memory_usage()

    for i in range(n_chunks):
        start_i = i * chunk_size
        end_i = min(start_i + chunk_size, n_samples)

        if end_i - start_i > 0:
          chunk_similarities = cosine_similarity( matrix[start_i:end_i], matrix )

          similarities[start_i:end_i] = chunk_similarities

          # Free memory
          del chunk_similarities
          gc.collect()

          # Progress and memory tracking
          current_memory = get_memory_usage()
          print(f"Chunk {i+1}/{n_chunks} complete. "
                f"Progress: {end_i}/{n_samples} rows. "
                f"Memory usage: {current_memory:.2f} MB "
                f"(Δ: {current_memory - initial_memory:.2f} MB)")
        else:
            print(f"Chunk {i+1}/{n_chunks} is empty, skipping...")

    return similarities

# Usage
chunk_size = 1000  # Adjust based on your RAM
cosine_sim = calculate_similarity_in_chunks(sample_genres, chunk_size)


movies = pd.read_csv('data/movies.csv')  # Load your movies DataFrame


# Load the models
# Load the models
@st.cache_resource
def load_trained_models():
    # Load the generator model for GAN
    generator_model = tf.keras.models.load_model(r'models/gan_generator_model.h5')
    
    # Load the discriminator model (if needed)
    discriminator_model = tf.keras.models.load_model(r'models/gan_discriminator_model.h5')
    
    # Load the SVD model using joblib
    svd_model = joblib.load(r'models/svdpp.pkl')

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
                svd_recommendations = get_svd_recommendations(user_id, svd_model)  # Collaborative filtering
                content_recommendations = get_content_based_recommendations(fav_movie, genres)
                gan_recommendations = get_gan_recommendations(generator_model, movies_df=df)

                # Display results
                 st.subheader("Collaborative Filtering Recommendations")
                 st.write(svd_recommendations)
                st.subheader("Content-Based Filtering Recommendations")
                st.write(content_recommendations)

                st.subheader("GAN-Based Recommendations")
                st.write(gan_recommendations)

                # Log recommendations as artifacts
                mlflow.log_artifact("Content Recommendations", content_recommendations)
                mlflow.log_artifact("GAN Recommendations", gan_recommendations)
                mlflow.log_artifact("Collaborative filtering Recommendations", svd_recommendations)

            else:
                st.write("Please provide both your favorite movie and genres.")

if __name__ == '__main__':
    main()
