import urllib.request
import json
import base64
import os

def download_dataset():
    print("Searching for space_missions.csv...")
    req = urllib.request.Request(
        'https://api.github.com/search/code?q="Status+Mission"+filename:space_missions.csv',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    try:
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read().decode())
            for item in data.get('items', []):
                raw_url = item['html_url'].replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
                print('Found raw URL:', raw_url)
                
                req2 = urllib.request.Request(raw_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req2) as r2:
                    content = r2.read()
                    os.makedirs('data/raw', exist_ok=True)
                    with open('data/raw/space_missions_large.csv', 'wb') as f:
                        f.write(content)
                print('Downloaded successfully!')
                return
    except Exception as e:
        print('Error:', e)
        
if __name__ == '__main__':
    download_dataset()
