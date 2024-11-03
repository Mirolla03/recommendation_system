import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from fuzzywuzzy import process
import mlflow
import mlflow.tensorflow
import mlflow.pyfunc
import os
from sklearn.metrics.pairwise import cosine_similarity
import gc
import psutil

def get_svdpp_recommendations(user_id, model, rating_df, movies_df, top_n=5):
    all_movie_ids = movies_df['movieId'].unique()
    watched_movie_ids = rating_df[rating_df['userId'] == user_id]['movieId'].unique()
    not_watched_ids = [mid for mid in all_movie_ids if mid not in watched_movie_ids]
    
    # Ensure model prediction is handled correctly
    predicted_ratings = [model.predict(user_id, mid) for mid in not_watched_ids]
    predicted_ratings.sort(key=lambda x: x.est, reverse=True)
    top_movie_ids = [pred.iid for pred in predicted_ratings[:top_n]]
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
            chunk_similarities = cosine_similarity(matrix[start_i:end_i], matrix)
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

# Load the preprocessed movie dataset
movies = pd.read_csv('data/movies.csv')  # Ensure the path is correct
df = pd.read_pickle("encoded_genres.pkl")  # Ensure the path is correct

@st.cache_resource
def load_trained_models():
    generator_model = tf.keras.models.load_model(r'models/gan_generator_model.h5')
    discriminator_model = tf.keras.models.load_model(r'models/gan_discriminator_model.h5')
    svd_model = joblib.load(r'models/svdpp.pkl')
    return generator_model, discriminator_model, svd_model

generator, discriminator, svd_model = load_trained_models()

def generate_synthetic_profile(latent_dim=35):
    noise = np.random.normal(0, 1, (1, latent_dim))
    synthetic_profile = generator.predict(noise)
    return synthetic_profile

def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

def filter_movies_by_genres(selected_genres):
    filtered_movies = movies[movies['genres'].apply(lambda x: any(genre in x for genre in selected_genres))]
    return filtered_movies

def get_content_based_recommendations(title_string, selected_genres, n_recommendations=10):
    title = movie_finder(title_string)
    movie_idx = dict(zip(movies['title'], list(movies.index)))
    idx = movie_idx[title]
    filtered_movies = filter_movies_by_genres(selected_genres)
    filtered_movie_indices = filtered_movies.index.tolist()
    sim_scores = [(i, cosine_sim[idx][i]) for i in filtered_movie_indices]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations + 1)]
    similar_movies = [i[0] for i in sim_scores]

    return movies['title'].iloc[similar_movies]

def get_gan_recommendations(generator_model, movies_df, top_n=5):
    generated_user_profile = generator_model.predict(np.random.randn(1, 100))
    recommended_movies = movies_df.sample(n=top_n)['title'].tolist()
    return recommended_movies

def main():
    st.title("Movie Recommendation System")

    with mlflow.start_run(run_name="movie_recommendation_experiment"):
        movie_list = df['title'].unique()
        fav_movie = st.selectbox("What is your favorite movie?", movie_list)
        genre_columns = [col for col in df.columns if col not in ['title', 'userId', 'rating', 'movie_id']]
        genres = st.multiselect("Select your top 3 favorite genres:", genre_columns)

        st.write(f"Favorite Movie: {fav_movie}")
        st.write(f"Selected Genres: {genres}")

        mlflow.log_param("favorite_movie", fav_movie)
        mlflow.log_param("selected_genres", genres)

        if st.button('Generate Synthetic Profile'):
            profile = generate_synthetic_profile()
            st.write('Generated Synthetic Profile:', profile)

            # Convert profile to a string or save to file to log in MLflow
            profile_filename = "generated_profile.npy"
            np.save(profile_filename, profile)
            mlflow.log_artifact(profile_filename)

        if st.button("Get Personalized Recommendations"):
            if fav_movie and genres:
                st.write(f"Your favorite movie is: {fav_movie}")
                st.write(f"Your favorite genres are: {', '.join(genres)}")

                # Assume user_id is predefined, or you can ask for user input
                user_id = 1  # Replace with actual user ID input method if necessary

                # Call recommendation functions
                svd_recommendations = get_svdpp_recommendations(user_id, svd_model, df, movies)  # Pass rating DataFrame
                content_recommendations = get_content_based_recommendations(fav_movie, genres)
                gan_recommendations = get_gan_recommendations(generator, movies_df=movies)

                st.subheader("Collaborative Filtering Recommendations")
                st.write(svd_recommendations)
                st.subheader("Content-Based Filtering Recommendations")
                st.write(content_recommendations)
                st.subheader("GAN-Based Recommendations")
                st.write(gan_recommendations)

                # Log recommendations as artifacts (convert lists to strings)
                mlflow.log_artifact("content_recommendations.txt", "\n".join(content_recommendations))
                mlflow.log_artifact("gan_recommendations.txt", "\n".join(gan_recommendations))
                mlflow.log_artifact("svd_recommendations.txt", "\n".join(svd_recommendations))

            else:
                st.write("Please provide both your favorite movie and genres.")

if __name__ == '__main__':
    main()
