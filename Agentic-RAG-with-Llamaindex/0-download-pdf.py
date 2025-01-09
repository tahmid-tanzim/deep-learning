import requests

url = "https://openreview.net/pdf?id=VtmBAGCN7o"
# url = "https://arxiv.org/pdf/2201.11903v6"
chunk_size = 2000  # bytes
r = requests.get(url, stream=True)

with open('./resource/metagpt.pdf', 'wb') as fd:
    for chunk in r.iter_content(chunk_size):
        fd.write(chunk)
