# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 08:12:38 2024

@author: sreev
"""
from pathlib import Path
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

user_art_matrix=load_user_art(Path("./data/user_artists.dat"))
print(user_art_matrix)
'''
artist_retriever = ArtistRetriever()
artist_retriever.load_artists(Path("./data/artists.dat"))
artist = artist_retriever.get_artist_name_from_id(1)
print(artist)
'''