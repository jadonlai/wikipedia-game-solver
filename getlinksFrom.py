import requests

# GETS THE LIST OF LINKS FROM A WEBPAGE. CHANGE THE GIVEN LINE

def query(request):
    request['action'] = 'query'
    request['format'] = 'json'
    # CHANGE THIS LINE
    request['titles'] = "Miss Meyers"
    request["prop"] = "linkshere"
    lastContinue = {}
    while True:
        # Clone original request
        req = request.copy()
        # Modify it with the values returned in the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = requests.get('https://en.wikipedia.org/w/api.php', params=req).json()
        if 'error' in result:
            raise Exception(result['error'])
        if 'warnings' in result:
            print(result['warnings'])
        if 'query' in result:
            yield result['query']
        if 'continue' not in result:
            break
        lastContinue = result['continue']

for result in query({}):
    reskey = result["pages"].keys()
    lists = result["pages"][list(reskey)[0]]["linkshere"]
    for j in lists:
        # The links with : are not allowed.
        if (":" not in j["title"]):
            print(j["title"])
