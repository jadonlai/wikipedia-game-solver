# 480_Wikipedia

Final Project
CSC 480, Cal Poly
Professor Canaan

Modules:
requests, sentence_transformers, plotly, kaleido
Optional Modules:
nltk, gensim, matplotlib

Install a module:
pip3 install module_name

Make the executable:
ln -s wikiSearch.py wikiSearch
chmod u+x wikiSearch

Usage:
python3 wikiSearch start end algorithm [verbosity]

Example: (3_6_24 folder, A*)
python3 wikiSearch 'Red algae' 'Wolfgang Amadeus Mozart' astar 2
Outputs a csv file with the results and a graph of cosine similarity vs. time if it succeeds

Extra Details:
- Used MAX_TIME = 36000 and MAX_NODES = 10000 for 3_5/6_24 and astar>gbfs_example
- Used MAX_TIME = 900 and MAX_NODES = 200 for all else
