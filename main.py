#!/usr/bin/python3

import requests



# Given a title, query the API and get the list of links from the webpage
def query(title):
    request = {'action':'query', 'format':'json', 'titles':title, 'prop':'links'}
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



def print_links(title):
    for result in query(title):
        reskey = result['pages'].keys()
        lists = result['pages'][list(reskey)[0]]['links']
        for j in lists:
            # The links with : are not allowed.
            if (':' not in j['title']):
                print(j['title'])



print_links('Miss Meyers')


# Format of the query completing.
format = {'pages':
          {'22822937':
           {'pageid':22822937, 'ns':0, 'title':'Miss Meyers',
            'links':[{'ns': 0, 'title': 'Speed index'},
                     {'ns': 0, 'title': 'Stakes race'},
                     {'ns': 0, 'title': 'Stallion'},
                     {'ns': 0, 'title': 'Thoroughbred'},
                     {'ns': 0, 'title': 'Three Bars'},
                     {'ns': 4, 'title': 'Wikipedia:Contents/Portals'},
                     {'ns': 4, 'title': 'Wikipedia:Featured articles'},
                     {'ns': 10, 'title': 'Template:Inflation/US'},
                     {'ns': 14, 'title': 'Category:Use American English from July 2017'},
                     {'ns': 14, 'title': 'Category:Use mdy dates from July 2021'}]}}}
