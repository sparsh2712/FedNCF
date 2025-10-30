import numpy as np
import os
from pathlib import Path

class MovieLensDataset:
    def __init__(self, ratings_path=Path(__file__).parent.parent / 'ml-1m' / 'ratings.dat', cache_path=Path(__file__).parent.parent / 'ml-1m' / 'ratings.npy'):
        self.ratings_path = ratings_path
        self.cache_path = cache_path
        self.rating_tuples = self._read_ratings()
        self.num_users = int(self.rating_tuples[:, 0].max())
        self.num_movies = int(self.rating_tuples[:, 1].max())
        self.ratings = self._load_or_generate_matrix()

    def _read_ratings(self):
        with open(self.ratings_path, 'r') as f:
            data = [line.strip().split('::')[:3] for line in f]
        return np.array(data, dtype=int)

    def _generate_matrix(self):
        matrix = np.zeros((self.num_users, self.num_movies), dtype=np.uint8)
        user_ids = self.rating_tuples[:, 0] - 1
        movie_ids = self.rating_tuples[:, 1] - 1
        ratings = self.rating_tuples[:, 2]
        matrix[user_ids, movie_ids] = ratings
        return matrix

    def _load_or_generate_matrix(self):
        if os.path.exists(self.cache_path):
            return np.load(self.cache_path)
        matrix = self._generate_matrix()
        np.save(self.cache_path, matrix)
        return matrix


if __name__ == '__main__':
    dataset = MovieLensDataset()
    print(dataset.ratings)
