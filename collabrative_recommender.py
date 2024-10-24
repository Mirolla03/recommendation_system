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
