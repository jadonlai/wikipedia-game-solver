#!/usr/bin/python3

"""
    get_links.py

    MediaWiki API Demos
    Demo of `Links` module: Get all links on the given page(s)

    MIT License
"""

import requests

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    "action": "query",
    "format": "json",
    "titles": "Albert Einstein",
    "prop": "links",
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()
PAGES = DATA["query"]["pages"]
for k, v in PAGES.items():
    for l in v["links"]:
        print(l["title"])

"""
def query(request):
    request['action'] = 'query'
    request['format'] = 'json'
    lastContinue = {}
    while True:
        # Clone original request
        req = request.copy()
        # Modify it with the values returned in the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = request.get('https://en.wikipedia.org/w/api.php', params=req).json()
        if 'error' in result:
            raise Exception(result['error'])
        if 'warnings' in result:
            print(result['warnings'])
        if 'query' in result:
            yield result['query']
        if 'continue' not in result:
            break
        lastContinue = result['continue']


for result in query({"titles": "Albert Einstein", "prop": "links"}):
    print(result)
"""