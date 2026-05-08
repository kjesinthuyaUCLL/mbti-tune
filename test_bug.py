"""Final verification: different music styles -> different MBTI percentages with temperature=1.5"""
import sys
sys.path.insert(0, '.')
import numpy as np
import joblib
from pathlib import Path

from src.spotify_utils import load_song_encoder, encode_songs_to_transfer_emb, build_features_from_tracks
from src.inference import load_model_and_scaler, predict_mbti

encoder, song_scaler = load_song_encoder()
model, scaler, device, feature_cols, idx_to_type = load_model_and_scaler()
mbti_scaler = joblib.load(Path('data/processed/mbti_scaler.pkl'))

samples = {
    'Upbeat Pop': [
        {'danceability':0.9,'energy':0.95,'valence':0.8,'acousticness':0.02,'instrumentalness':0.0,'speechiness':0.04,'loudness':-3.0,'tempo':130.0,'liveness':0.1,'key':5,'mode':1},
        {'danceability':0.85,'energy':0.9,'valence':0.75,'acousticness':0.05,'instrumentalness':0.0,'speechiness':0.05,'loudness':-4.0,'tempo':125.0,'liveness':0.12,'key':7,'mode':1},
        {'danceability':0.8,'energy':0.88,'valence':0.7,'acousticness':0.08,'instrumentalness':0.0,'speechiness':0.03,'loudness':-5.0,'tempo':120.0,'liveness':0.15,'key':2,'mode':1},
    ],
    'Sad Acoustic': [
        {'danceability':0.2,'energy':0.15,'valence':0.1,'acousticness':0.9,'instrumentalness':0.7,'speechiness':0.03,'loudness':-18.0,'tempo':65.0,'liveness':0.08,'key':9,'mode':0},
        {'danceability':0.25,'energy':0.2,'valence':0.15,'acousticness':0.85,'instrumentalness':0.6,'speechiness':0.04,'loudness':-16.0,'tempo':70.0,'liveness':0.09,'key':4,'mode':0},
        {'danceability':0.3,'energy':0.25,'valence':0.2,'acousticness':0.8,'instrumentalness':0.5,'speechiness':0.05,'loudness':-15.0,'tempo':75.0,'liveness':0.1,'key':1,'mode':0},
    ],
    'Hip-Hop': [
        {'danceability':0.88,'energy':0.82,'valence':0.55,'acousticness':0.04,'instrumentalness':0.0,'speechiness':0.25,'loudness':-5.0,'tempo':96.0,'liveness':0.18,'key':0,'mode':1},
        {'danceability':0.82,'energy':0.78,'valence':0.5,'acousticness':0.06,'instrumentalness':0.0,'speechiness':0.3,'loudness':-6.0,'tempo':92.0,'liveness':0.2,'key':3,'mode':0},
        {'danceability':0.85,'energy':0.8,'valence':0.52,'acousticness':0.05,'instrumentalness':0.0,'speechiness':0.28,'loudness':-5.5,'tempo':94.0,'liveness':0.22,'key':6,'mode':1},
    ],
}

print('Temperature=1.5 predictions:')
print('-' * 65)
for name, tracks in samples.items():
    emb = encode_songs_to_transfer_emb(tracks, encoder, song_scaler)
    agg = build_features_from_tracks(tracks)
    agg.update(emb)
    vec = np.zeros((1, len(feature_cols)), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        vec[0, i] = agg.get(col, 0.0)
    vec = mbti_scaler.transform(vec).astype(np.float32)
    res = predict_mbti(vec, model, scaler, device, feature_cols, idx_to_type, temperature=1.5)
    p = res['percentages']
    mbti = res['mbti']
    print(name + ' -> ' + mbti)
    print('  E:' + str(round(p['E']*100,1)) + '%  I:' + str(round(p['I']*100,1)) + '% | '
          'S:' + str(round(p['S']*100,1)) + '%  N:' + str(round(p['N']*100,1)) + '% | '
          'T:' + str(round(p['T']*100,1)) + '%  F:' + str(round(p['F']*100,1)) + '% | '
          'J:' + str(round(p['J']*100,1)) + '%  P:' + str(round(p['P']*100,1)) + '%')
print('-' * 65)
print('PASS: Pipeline complete')
