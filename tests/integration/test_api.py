import requests


BASE_URL = "http://localhost:8887"


def test_api_is_running():
    r = requests.get(f"{BASE_URL}/api")
    assert r.status_code == 200


def test_predict():
    r = requests.post(
        f"{BASE_URL}/predict",
        json={
            "classifier_type": "random_forest",
            "hyperparameters": {},
            "NEds": 5,
            "NActDays": 10,
            "pagesWomen": 3,
            "wikiprojWomen": 4,
            "additional_data": []
        })
    assert r.status_code == 200
    body = r.json()
    assert body["predictions"][0] == 2
