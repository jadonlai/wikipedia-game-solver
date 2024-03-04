#!/usr/bin/python3

import sys, requests, time, gensim, warnings
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, util
from queue import PriorityQueue
from matplotlib import pyplot as plt



MAX_NODES = 100000
MAX_TIME = 3600

v = 0



# Format of the query completing
format = {'pages':
          {'22822937':
           {'pageid':22822937, 'ns':0, 'title':'Miss Meyers',
            'links':[{'ns':0, 'title':'Speed index'},
                     {'ns':0, 'title':'Stakes race'},
                     {'ns':0, 'title':'Stallion'},
                     {'ns':0, 'title':'Thoroughbred'},
                     {'ns':0, 'title':'Three Bars'},
                     {'ns':4, 'title':'Wikipedia:Contents/Portals'},
                     {'ns':4, 'title':'Wikipedia:Featured articles'},
                     {'ns':10, 'title':'Template:Inflation/US'},
                     {'ns':14, 'title':'Category:Use American English from July 2017'},
                     {'ns':14, 'title':'Category:Use mdy dates from July 2021'}]}}}



# Given a title, query the API and get the list of links from the webpage
def query(title):
    request = {'action':'query', 'format':'json', 'titles':title, 'prop':'links'}
    lastContinue = {}
    while True:
        # Clone original request
        req = request.copy()
        # Modify it with the values returned in the 'continue' section of the last result
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



# Given a title, get the list of links or None if it doesn't exist
def get_links(title):
    # Init res list
    res = []
    # Get the query result
    for result in query(title):
        # Access the list of links
        reskey = result['pages'].keys()
        # No links
        if 'missing' in result['pages'][list(reskey)[0]]:
            return None
        lists = result['pages'][list(reskey)[0]]['links']
        # Append each link
        for link in lists:
            if (':' not in link['title']):
                res.append(link['title'])
    # No links
    if len(res) == 0:
        return None
    return res



# Given a title, print the links from that webpage
def print_links(title):
    for link in get_links(title):
        print(link)



# Given a list of times and a list of heuristics, plot the heuristics vs. time
def plot(times, heuristics):
    plt.plot(times, heuristics)



# Word2Vec version that will compute words based on their similarity in alphabetical order (doesn't work)
# def get_similarities(word, links):
#     # Setup the list of words
#     links.insert(0, word)
#     links = [[x] for x in links]
#     # Create model
#     model = gensim.models.Word2Vec(links, min_count=1, vector_size=len(links), window=5)
#     # Return list of words, sorted by the most similar
#     return model.wv.most_similar(word, topn=len(links) - 1)

# Given a word and list of links, return a sorted array of the most similar words as tuples
def get_similarities(word, links):
    res = []
    # Init model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute embedding for both word and links
    embeddings1 = model.encode([word], convert_to_tensor=True)
    embeddings2 = model.encode(links, convert_to_tensor=True)
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    # Add word and similarity to res list
    for i in range(len(links)):
        res.append([links[i], cosine_scores[0][i].item()])

    # Sort the res list by similarity
    res.sort(key=lambda x:x[1], reverse=True)
    
    return res



# Given a parent, start, and end, return the optimal path
def backtrace(parent, start, end):
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path



# Given a start and end goal, find the end goal from the start using BFS and return the optimal path, search dist, and time
# Return None if no path is found or if a path is too long
def bfs(start, end):
    # Start timer
    start_time = time.time()

    # Init variables
    dist = 0
    parent = {}
    parent[start] = None
    queue = []
    queue.append(start)

    # BFS
    while queue:
        # Run for max MAX_TIME
        if time.time() - start_time > MAX_TIME:
            print('Exceeded max time')
            return None
        
        # Pop the first element
        node = queue.pop(0)
        # Increment the dist
        dist += 1

        # Verbosity
        if v >= 1:
            print('Distance:', dist)
            print('Node:', node)
        if v == 2:
            print('Elapsed Time:', round(time.time() - start_time, 3))
        print()

        # Run for max MAX_NODES nodes
        if dist > MAX_NODES:
            print('Exceeded max nodes')
            return None
        
        # Check if the node is the end
        if node.lower() == end.lower():
            return backtrace(parent, start, node), dist, time.time() - start_time
        # Add adjacent nodes to the queue
        links = get_links(node)
        # Continue to the next link if no children
        if links == None:
            continue
        for adjacent in links:
            if adjacent not in parent:
                parent[adjacent] = node
                queue.append(adjacent)
    return None



# UNFINISHED
def dfs(visited, node, end):
    return None



# Given a start and end goal, find the end goal from the start using GBFS or A* and return the path, search dist, and time
# Return None if no path is found or if a path is too long
def gbfs_astar(algorithm, start, end):
    # Start timer
    start_time = time.time()

    # Init variables
    dist = 0
    parent = {}
    parent[start] = None
    visited = [False] * MAX_NODES
    pq = PriorityQueue()
    pq.put((0, start))
    visited = set()
    nodes = []
    heuristics = []
    times = []
    fig, ax = plt.subplots()

    # GBFS or A*
    while pq.empty() == False:
        # Run for max MAX_TIME
        if time.time() - start_time > MAX_TIME:
            print('Exceeded max time')
            return None
        
        # Get the most similar link
        sim, node = pq.get()
        # Don't revisit a node
        if node in visited:
            continue
        # Increment the dist
        dist += 1

        # Verbosity
        if v >= 1:
            print('Distance:', dist)
            print('Node:', node)
        if v == 2:
            # GBFS
            if algorithm == 'gbfs':
                print('Heuristic: ', sim)
            # A*
            else:
                print('Cost + Heuristic:', sim)
            print('Elapsed Time:', round(time.time() - start_time, 3))
        print()

        # Graph
        nodes.append(node)
        heuristics.append(sim)
        times.append(time.time() - start_time)

        # Run for max MAX_NODES nodes
        if dist > MAX_NODES:
            print('Exceeded max nodes')
            return None
        
        # Add node to visited
        visited.add(node)
        # Check if node is the end
        if node.lower() == end.lower():
            # Plot times vs. heuristics
            if (v >= 1):
                ax.plot(times, heuristics, marker='o')
                for i in range(len(times)):
                    ax.text(times[i], heuristics[i], nodes[i], fontsize=5)
                plt.savefig(f'{start}_{end}_{algorithm}_plot.png')
            # Return path
            return backtrace(parent, start, node), dist, time.time() - start_time
        # Get children
        links = get_links(node)
        # Continue to the next link if no children
        if links == None:
            continue
        similarities = get_similarities(end, links)
        # Add adjacent nodes to priority queue
        for link, similarity in similarities:
            if link not in visited:
                parent[link] = node
                # GBFS
                if algorithm == 'gbfs':
                    pq.put((1 - similarity, link))
                # A*
                else:
                    pq.put((sim + 1 - similarity, link))



def machine_learning():
    pass



if __name__ == '__main__':
    # Parse args
    args = sys.argv[1:]
    if len(args) > 4 or len(args) == 0:
        print('Usage: python3 wikiSearch start end algorithm [verbosity]')
        exit(1)
    start = args[0]
    end = args[1]
    algorithm = args[2]
    if algorithm not in ['bfs', 'dfs', 'gbfs', 'astar', 'machine_learning']:
        print('Invalid algorithm')
        exit(1)
    if len(args) == 4:
        if args[3].isdigit() and int(args[3]) >= 0 and int(args[3]) <= 2:
            v = int(args[3])
    
    # TEST
    # Start = Deodorant
    # End = Parachute
    # Optimal is 3 steps (Deodorant -> Inventor -> Parachute)

    if algorithm == 'bfs':
        print(bfs(start, end))
    elif algorithm == 'dfs':
        print(dfs(start, end))
    elif algorithm == 'gbfs' or algorithm == 'astar':
        print(gbfs_astar(algorithm, start, end))
    elif algorithm == 'machine_learning':
        print(machine_learning(start, end))
