import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Product

def recommend_products(product_id, num_recommendations=5):
    # Get the products from the database
    products = Product.objects.all()
    products_df = pd.DataFrame(products.values('id', 'name', 'description'))

    # Initialize the TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    # Fit the model and transform the product descriptions
    tfidf_matrix = tfidf.fit_transform(products_df['description'])

    # Calculate cosine similarities between products
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index of the current product
    idx = products_df[products_df['id'] == product_id].index[0]

    # Get the similarity scores for the product with the given ID
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products by similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the most similar products
    product_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

    # Return the recommended products
    recommended_products = products_df.iloc[product_indices]
    return recommended_products