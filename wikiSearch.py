#!/usr/bin/python3

import sys, requests, time, gensim, warnings, csv
import plotly.express as px
# from gensim.models import Word2Vec
# from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, util
from queue import PriorityQueue
# from matplotlib import pyplot as plt



MAX_NODES = 10000
MAX_TIME = 36000

v = 0



'''
Format of the query completing
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
'''



# Class to represent a link and its information
class Link:
    def __init__(self, name, parent, g, h, level):
        self.name = name
        self.parent = parent
        self.g = g
        self.h = h
        self.f = self.g + self.h
        self.level = level

    # Compare links based on their f values
    def __lt__(self, other):
        return self.f < other.f

    # Calculate the f value
    def set_f(self):
        self.f = self.g + self.h



# Given a list of indices, return a list of positions that overlap less
def improve_text_position(x):
    positions = ['top center', 'bottom center', 'middle left', 'middle right']
    return [positions[i % len(positions)] for i in range(len(x))]



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



# Given a 2D array, start, end, and algorithm, save it to a csv file
def save_to_csv(array, start, end, algorithm):
    with open(f'{start}_{end}_{algorithm}.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerows(array)



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
def backtrace(start, end):
    cur = end
    path = []
    while cur.name != start:
        path.append(cur.name)
        cur = cur.parent
    path.append(start)
    path.reverse()
    return path



# Given a start and end goal,
# find the end goal from the start using BFS and return the optimal path, search dist, and time
# Exit if no path is found or if a path is too long
def bfs(start, end):
    # Start timer
    start_time = time.time()

    # Init variables
    dist = -1
    visited = set()
    queue = []
    queue.append(Link(start, None, 0, 0, 0))
    res = [['Node', 'Level', 'Queue Size', 'Distance', 'Elapsed Time']]

    # BFS
    while queue:
        # Run for max MAX_TIME
        t = round(time.time() - start_time, 3)
        if t > MAX_TIME:
            print('Exceeded max time')
            # Updated res csv
            res.append([])
            res.append(['Exceeded max time'])
            save_to_csv(res, start, end, 'bfs')
            sys.exit()
        
        # Run for max MAX_NODES nodes
        if dist > MAX_NODES:
            print('Exceeded max nodes')
            # Update res csv
            res.append([])
            res.append(['Exceeded max nodes'])
            save_to_csv(res, start, end, 'bfs')
            sys.exit()
        
        # Pop the first element
        node = queue.pop(0)
        # Increment the dist
        dist += 1

        # Verbosity
        if v >= 1:
            print('Node:', node.name)
        if v == 2:
            print('Level:', node.level)
            print('Queue Size:', len(queue))
            print('Distance:', dist)
            print('Elapsed Time:', t)
        if v != 0:
            print()

        # Add details to res list
        res.append([node.name, node.level, len(queue), dist, t])
        
        # Check if the node is the end
        if node.name.lower() == end.lower():
            # Update res csv
            res.append([])
            res.append([backtrace(start, node)])
            save_to_csv(res, start, end, 'bfs')
            # Return path
            return backtrace(start, node), dist, t
        # Add adjacent nodes to the queue
        links = get_links(node.name)
        # Continue to the next link if no children
        if links == None:
            continue
        for adjacent in links:
            if adjacent not in visited:
                visited.add(adjacent)
                queue.append(Link(adjacent, node, 0, 0, node.level + 1))



# Given a node, start, end goal, parent dict, search dist, start time, and res list,
# find the end goal from the start using DFS and return the optimal path, search dist, and time
# Exit if no path is found or if a path is too long
def dfs(node, start, end, visited=None, dist=0, start_time=time.time(), res = [['Node', 'Level', 'Distance', 'Elapsed Time']]):
    # Start of the algorithm
    if visited is None:
        visited = set()
        visited.add(node.name)

    # Run for max MAX_TIME
    t = round(time.time() - start_time, 3)
    if t > MAX_TIME:
        print('Exceeded max time')
        # Update res csv
        res.append([])
        res.append(['Exceeded max time'])
        save_to_csv(res, start, end, 'dfs')
        sys.exit()

    # Run for max MAX_NODES nodes
    if dist > MAX_NODES:
        print('Exceeded max nodes')
        # Update res csv
        res.append([])
        res.append(['Exceeded max nodes'])
        save_to_csv(res, start, end, 'dfs')
        sys.exit()

    # Verbosity
    if v >= 1:
        print('Node:', node.name)
    if v == 2:
        print('Level:', node.level)
        print('Distance:', dist)
        print('Elapsed Time:', t)
    if v != 0:
        print()

    # Add details to res list
    res.append([node.name, node.level, dist, t])

    # Check if the node is the end
    if node.name.lower() == end.lower():
        # Update res csv
        res.append([])
        res.append([backtrace(start, node)])
        save_to_csv(res, start, end, 'dfs')
        # Return path
        return backtrace(start, node), dist, t
    # Add adjacent nodes to the queue
    links = get_links(node.name)
    # Continue to the next link if no children
    if links == None:
        return
    for adjacent in links:
        if adjacent not in visited:
            visited.add(adjacent)
            dfs(Link(adjacent, node, 0, 0, node.level + 1), start, end, visited, dist + 1, start_time)



# Given an algorithm, start, and end goal,
# find the end goal from the start using GBFS or A* and return the path, search dist, and time
# Exit if no path is found or if a path is too long
def gbfs_astar(algorithm, start, end):
    # Start timer
    start_time = time.time()

    # Init variables
    dist = -1
    visited = [False] * MAX_NODES
    pq = PriorityQueue()
    pq.put(Link(start, None, 0, 0, 0))
    visited = set()
    nodes = []
    heuristics = []
    times = []
    res = [['Node', 'Level', 'Total Cost', 'Priority Queue Size', 'Distance', 'Elapsed Time']]
    # Matplotlib
    # fig, ax = plt.subplots()

    # GBFS or A*
    while pq.empty() == False:
        # Run for max MAX_TIME
        t = round(time.time() - start_time, 3)
        if t > MAX_TIME:
            print('Exceeded max time')
            # Update res csv
            res.append([])
            res.append(['Exceeded max time'])
            save_to_csv(res, start, end, algorithm)
            sys.exit()
        
        # Run for max MAX_NODES nodes
        if dist > MAX_NODES:
            print('Exceeded max nodes')
            # Update res csv
            res.append([])
            res.append(['Exceeded max nodes'])
            save_to_csv(res, start, end, algorithm)
            sys.exit()
        
        # Get the most similar link
        node = pq.get()
        # Don't revisit a node
        if node.name in visited:
            continue
        # Increment the dist
        dist += 1

        # Verbosity
        if v >= 1:
            print('Node:', node.name)
        if v == 2:
            print('Level:', node.level)
            # GBFS
            if algorithm == 'gbfs':
                print('Heuristic:', node.h)
            # A*
            else:
                print('Cost + Heuristic:', node.f)
            print('Priority Queue Size:', pq.qsize())
            print('Distance:', dist)
            print('Elapsed Time:', t)
        if v != 0:
            print()

        # Add details to res list
        res.append([node.name, node.level, node.f, pq.qsize(), dist, t])

        # Graph
        nodes.append(node.name)
        heuristics.append(node.f)
        times.append(t)
        
        # Add node to visited
        visited.add(node.name)
        # Check if node is the end
        if node.name.lower() == end.lower():
            # Plot times vs. heuristics
            if (v == 2):
                # Matplotlib
                # ax.plot(times, heuristics, marker='o')
                # for i in range(len(times)):
                #     ax.text(times[i], heuristics[i], nodes[i], fontsize=5)

                # plt.savefig(f'{start}_{end}_{algorithm}_plot.png')

                # Plotly Express
                fig = px.line(x=times, y=heuristics, markers=True, text=nodes)
                fig.update_traces(textposition=improve_text_position(range(len(nodes))), textfont_size=10)
                fig.write_image(f'{start}_{end}_{algorithm}_plot.png')
            # Update res csv
            res.append([])
            res.append([backtrace(start, node)])
            save_to_csv(res, start, end, algorithm)
            # Return path
            return backtrace(start, node), dist, t
        # Get children
        links = get_links(node.name)
        # Continue to the next link if no children
        if links == None:
            continue
        similarities = get_similarities(end, links)
        # Add adjacent nodes to priority queue
        for link, similarity in similarities:
            if link not in visited:
                # Set link class
                link = Link(link, node, 0, 1 - similarity, node.level + 1)
                # A* includes cost
                if algorithm == 'astar':
                    link.g = node.f
                # Set f and put the link into the queue
                link.set_f()
                pq.put(link)



if __name__ == '__main__':
    # Parse args
    args = sys.argv[1:]
    if len(args) > 4 or len(args) == 0:
        print('Usage: python3 wikiSearch start end algorithm [verbosity]')
        exit(1)
    start = args[0]
    end = args[1]
    algorithm = args[2]
    if algorithm not in ['bfs', 'dfs', 'gbfs', 'astar']:
        print('Invalid algorithm')
        exit(1)
    if len(args) == 4:
        if args[3].isdigit() and int(args[3]) >= 0 and int(args[3]) <= 2:
            v = int(args[3])
    
    # TEST SAMPLE
    # Start = Deodorant
    # End = Parachute
    # Optimal is 3 steps (Deodorant -> Inventor -> Parachute)

    if algorithm == 'bfs':
        print(bfs(start, end))
    elif algorithm == 'dfs':
        print(dfs(Link(start, 0, 0, 0), start, end))
    elif algorithm == 'gbfs' or algorithm == 'astar':
        print(gbfs_astar(algorithm, start, end))
