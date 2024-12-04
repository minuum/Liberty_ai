import requests

def test_streamlit_server():
    response = requests.get("http://localhost:8501")
    assert response.status_code == 200
    assert "Streamlit" in response.text