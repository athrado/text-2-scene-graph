# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# load_conceptnet.py
# ----------------
# Load full ConceptNet data, reduce to relevant Concepts
# and save results as graph


# Imports
import csv
import re
import os
import pickle

from wordfreq import word_frequency, zipf_frequency
import networkx as nx

# Settings
MIN_ZIPF = 2.0
conceptnet_file = "./data/ConceptNet/conceptnet-assertions-5.7.0.csv"

# Load conceptnet and extract relevant concept triples
with open(conceptnet_file,'r') as csvfile:

	# Read conceptnet database
	read = csv.reader(csvfile, delimiter='\t', quotechar='\t')

	# Initalize
	nodes_1 = []
	nodes_2 = []
	relations = []
	rel_dict = {}
	n_original_triples = 0

	# Count exclusions
	excluded = 0
	total = 0

	# For each triple, check whether or not to save
	for line in read:
		
		# Count original triples
		n_original_triples += 1

		# Only use English concepts
		if '/c/en/' in line[2] and '/c/en/' in line[3]:

			# Select and clean first concept

			n1 = line[2]
			n1 = re.sub('/c/en','', n1)
			n1 = re.sub(r'/v$','', n1)
			n1 = re.sub(r'/a$','', n1)
			n1 = re.sub(r'/s$','', n1)
			n1 = re.sub(r'/r$','', n1)
			n1 = re.sub(r'/n$','', n1)
			n1 = re.sub(r'/en_\d+$','', n1)
			n1 = re.sub('/v/','/', n1)
			n1 = re.sub('/a/','/', n1)
			n1 = re.sub('/s/','/', n1)
			n1 = re.sub('/r/','/', n1)
			n1 = re.sub('/n/','/', n1)
			n1 = re.sub('/wikt/','/',n1)

			n1 = n1.split('/')[-1]

			# Select and clean second concept

			n2 = line[3]
			n2 = re.sub('/c/en','', n2)
			n2 = re.sub(r'/v$','', n2)
			n2 = re.sub(r'/a$','', n2)
			n2 = re.sub(r'/s$','', n2)
			n2 = re.sub(r'/r$','', n2)
			n2 = re.sub(r'/n$','', n2)
			n2 = re.sub(r'/en_\d+$','', n2)
			n2 = re.sub('/v/','/', n2)
			n2 = re.sub('/a/','/', n2)
			n2 = re.sub('/s/','/', n2)
			n2 = re.sub('/r/','/', n2)
			n2 = re.sub('/n/','/', n2)
			n2 = re.sub('/wikt/','/',n2)

			n2 = n2.split('/')[-1]

			# Replace _ with white string
			n1_corrected = re.sub('_',' ',n1)
			n2_corrected = re.sub('_',' ',n2)

			# Get zipf frequencies for concepts
			zipf_freq_n1 = zipf_frequency(n1_corrected,'en',wordlist='large')
			zipf_freq_n2 = zipf_frequency(n2_corrected,'en',wordlist='large')

			# Discard if zipf frequency not high enough
			if zipf_freq_n1 <= MIN_ZIPF:
				excluded += 1
				continue
			if zipf_freq_n2 <= MIN_ZIPF:
				excluded += 1
				continue

			total += 1

			# Append concepts to nodes
			nodes_1.append(n1)
			nodes_2.append(n2)

			# Get relation and append to list
			relation = line[1][3:]
			relations.append(relation)

			#print(n1)
			#print(n2)
			#print(relation)

			# Save concept1, concept2 and relation in dict
			if (n1,n2) in rel_dict.keys():
				rel_dict[(n1,n2)].append(relation)
			else:
				rel_dict[(n1,n2)] = [relation]


	print('LOADING CONCEPTNET ASSERTIONS\n===========================')
	print('# original triples:', n_original_triples)
	print('# excluded triples:', excluded)
	print('# remaining triples:', total)

	# Get all edges and unique nodes
	edges=list(zip(nodes_1,nodes_2))
	unique_nodes = list(set(nodes_1 + nodes_2))

	print('\n# edges:', len(edges))
	print('# unique nodes:', len(unique_nodes))

	# Create DiGraph
	G=nx.DiGraph()
	G.add_nodes_from(unique_nodes)
	G.add_edges_from(edges)

	# Save relation between concepts as edge label
	for (n1,n2) in G.edges:
		G[n1][n2]['label'] = rel_dict[(n1,n2)]

	# Save as undirected graph
	undir_G = G.to_undirected().copy()

	# Initialize directionality dictionary
	undir_G.dir_dict = {}

	# Create entry for all edges
	for (n1,n2) in edges:

		undir_G[n1][n2]['source node'] = []
		undir_G[n1][n2]['target node'] = []
		undir_G[n1][n2]['label'] = []


	# For each edge (concept1, concept2)
	for i, (node_1, node_2) in enumerate(undir_G.edges):

		# Get the node pairs (both directions)
		node_pairs = [(node_1,node_2),(node_2,node_1)]

		# For each pair, save relation and direction
		for (n1,n2) in node_pairs:

			# If this pair exists
			if (n1,n2) in rel_dict.keys():

				# Get the relations between two concepts as list
				relations = rel_dict[(n1,n2)]
				relations = list(set(relations))

				# For each relation
				for relation in relations:

					# Discard if this relation was already processed
					if (n1,n2,relation) in undir_G.dir_dict.keys() and undir_G.dir_dict[(n1,n2,relation)] != '<--':
						continue

					# Save source, target node and label
					undir_G[n1][n2]['source node'].append(n1)
					undir_G[n1][n2]['target node'].append(n2)
					undir_G[n1][n2]['label'].append(relation)

					# For bidirectional relations, save direction <-->
					if relation in ['RelatedTo','Synonym','Antonym','DistinctFrom','EtymologicallyRelatedTo']:
						undir_G.dir_dict[(n1,n2,relation)] = '<-->'
						undir_G.dir_dict[(n2,n1,relation)] = '<-->'

					# Otherwise, save direction
					else:
						undir_G.dir_dict[(n1,n2,relation)] = '-->'
						undir_G.dir_dict[(n2,n1,relation)] = '<--'


TAG = ''

# Save ConceptNet graph
nx.write_gpickle(undir_G, "data/ConceptNet/conceptnet_full_di_rel"+TAG+".gpickle")

# Save direction dict
with open('data/ConceptNet/dir_dict.pickle', 'wb') as handle:
	pickle.dump(undir_G.dir_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load graph, get all nodes
cn_graph_full = nx.read_gpickle('data/ConceptNet/conceptnet_full_di_rel'+TAG+'.gpickle')
cn_nodes = list(cn_graph_full.nodes)

print('# nodes:', len(cn_nodes))

# Discard nodes that have only one neighbor, create new subgraph and save
cn_nodes = [c for c in cn_nodes if len(list(cn_graph_full.neighbors(c)))>1]
reduced_graph = nx.subgraph(cn_graph_full, cn_nodes).copy()
nx.write_gpickle(reduced_graph,"data/ConceptNet/conceptnet_full_di_rel_red"+TAG+".gpickle")

print('# nodes (after removing nodes without neighbors):', len(cn_nodes))
