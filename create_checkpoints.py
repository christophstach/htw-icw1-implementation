import os
from typing import Dict

import requests
from dotenv import load_dotenv

load_dotenv()

from determined.experimental import Determined, ModelOrderBy


def create_checkpoints(trials: Dict[str, int]):
    master = os.getenv("DET_MASTER")
    login_data = {
        'username': os.getenv('DET_USERNAME'),
        'password': os.getenv('DET_PASSWORD')
    }
    login_response = requests.post(f'{master}/api/v1/auth/login', json=login_data).json()
    token = login_response['token']
    headers = {'Authorization': f'Bearer {token}'}

    assert token is not None

    for name, trial in trials.items():
        response = requests.get(f'{master}/api/v1/models/contrastive_model', headers=headers)
        print(response.text)

    data = []


create_checkpoints(
    {
        'depth 64': 26478,
        # 'depth 32': 26477,
        # 'depth 16': 25637,
        # 'depth 8': 25635
    }
)

d = Determined()

models = d.get_models()
model = d.get_model('gan_anime_face_128_8')
# model.register_version('ee004fe4-fd7b-458d-bff9-8859cebcda80')  # best
# model.register_version('222cb030-72a7-4952-9eb7-b997d65dab23')  # latest

print()
