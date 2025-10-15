import requests

def test_prediction():
    try:
        response = requests.get("http://127.0.0.1:8000/predict/AAPL")
        print("Response status:", response.status_code)
        print("Response data:", response.json())
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    test_prediction()