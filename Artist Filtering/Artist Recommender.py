# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 20:55:00 2024

@author: sreev
"""

from pathlib import Path
from typing import Tuple, List
import implicit
import scipy
import pandas as pd
from scipy.sparse import coo_matrix

def load_user_art(user_art_file: Path) -> scipy.sparse.csr_matrix:
    user_art = pd.read_csv(user_art_file, sep='\t')
    user_art.set_index(['userID', 'artistID'], inplace=True)
    coo = coo_matrix(
        (
            user_art.weight.astype(float),
            (
                user_art.index.get_level_values(0),
                user_art.index.get_level_values(1),
            ),
        )
    )
    return coo.tocsr()

class ArtistRetriever:
    
    def __init__(self):
        self._artist_df = None
        
    def get_artist_name_from_id(self, artist_id: int) -> str:
        return self._artist_df.loc[artist_id, 'name']
    
    def load_artists(self, art_file: Path) -> None:
        self._artist_df = pd.read_csv(art_file, delimiter='\t').set_index('id')

class ImplicitRecommender:
    def __init__(self, artist_retriever: ArtistRetriever,
                 implicit_model: implicit.recommender_base.RecommenderBase):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        self.implicit_model.fit(user_artists_matrix)

    def recommend(self, user_id: int,
                  user_artists_matrix: scipy.sparse.csr_matrix,
                  n: int = 10) -> Tuple[List[str], List[float]]:
        artist_ids, scores = self.implicit_model.recommend(user_id, user_artists_matrix)
        artists = [self.artist_retriever.get_artist_name_from_id(artist_id)
                   for artist_id in artist_ids]
        return artists, scores

user_artists = load_user_art(Path("./data/user_artists.dat"))

artist_retriever = ArtistRetriever()
artist_retriever.load_artists(Path("./data/artists.dat"))

implicit_model = implicit.als.AlternatingLeastSquares(
    factors=50, iterations=10, regularization=0.01)

recommender = ImplicitRecommender(artist_retriever, implicit_model)
recommender.fit(user_artists)
artists, scores = recommender.recommend(2, user_artists, n=5)

for artist, score in zip(artists, scores):
    print(f"{artist}, {score}")