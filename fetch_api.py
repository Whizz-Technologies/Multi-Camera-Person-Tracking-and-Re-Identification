import requests

def get_path(p):
    url = "http://127.0.0.1:5000/runpy"
    payload = {'path': p}
    print(payload)
    r = requests.post(url, data=payload)
    return True


if __name__ == '__main__':
    p = "./video/"
    print(get_path(p))