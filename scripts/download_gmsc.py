"""Download the public GMSC mirror to ignored local data/, verifying this run's pinned hashes."""
import hashlib
import json
import urllib.request
from pathlib import Path
from src.discover.discover_agent import atomic_json, now

FILES = [{'name': 'Data Dictionary.xls', 'url': 'https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master/3.%20Kaggle/Give%20Me%20Some%20Credit/Data%20Dictionary.xls', 'sha256': 'd388d6b11ed63fee95432048187c8c064b8ca38e40d7e8aab9edd36af240b092', 'bytes': 14848, 'mirror_blob_sha': 'a3d185aaaa02f354dac6cc5f4bc333fbd34b32ec'}, {'name': 'cs-test.csv', 'url': 'https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master/3.%20Kaggle/Give%20Me%20Some%20Credit/cs-test.csv', 'sha256': 'bab363a2a807218d32a51f5fc9668b8be7977795065edd386abc8546abaa5b78', 'bytes': 4983329, 'mirror_blob_sha': 'edaccfeaf75c16124ce6f95c989802485c23e244'}, {'name': 'cs-training.csv', 'url': 'https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master/3.%20Kaggle/Give%20Me%20Some%20Credit/cs-training.csv', 'sha256': '1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac', 'bytes': 7564965, 'mirror_blob_sha': '6e670f83b7788e0e57cc8a7719c70dcf2c071f3f'}, {'name': 'sampleEntry.csv', 'url': 'https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master/3.%20Kaggle/Give%20Me%20Some%20Credit/sampleEntry.csv', 'sha256': '578b4b01d0f6ed7f97f1988afff0c41194e72bf4b25707119903d4de3dc1dcea', 'bytes': 1907226, 'mirror_blob_sha': '18feb5c7b3de826463e58a685b57e4d9091f3883'}]

def main():
    root=Path('data/GiveMeSomeCredit');root.mkdir(parents=True,exist_ok=True)
    for item in FILES:
        path=root/item['name']
        data=path.read_bytes() if path.exists() else urllib.request.urlopen(item['url'],timeout=120).read()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:
            raise ValueError('Mirror bytes changed: '+item['name'])
        path.write_bytes(data)
        print(item['name']+' verified')
    atomic_json(root/'download_manifest.json',{'retrieved_at':now(),
      'original':'https://www.kaggle.com/competitions/GiveMeSomeCredit/data',
      'source_kind':'Public third-party mirror; original Kaggle bytes not independently authenticated',
      'files':FILES})

if __name__=='__main__':main()
