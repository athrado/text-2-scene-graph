# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# get_pageranks.py
# ----------------
# Compute the personalized PageRanks for scene graph and phrase concepts
# given a subgraph for image/caption pair


# Import statements
import os
import sys
import ast
import pickle

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

import conceptnet
import Scene_Graph as sg
import concept_config as cfg

from dask.distributed import Client
from dask_jobqueue import SLURMCluster



def get_pagerank(line, check_number=True):
	"""Get personalized PageRank for each scene graph and phrase concept pair as matrix."""

	# Load ConceptNet data
	cn_graph_full, dir_dict = conceptnet.load_conceptnet_data()

	# Get IDs
	ID, concepts = line.split('\t')
	image_ID = ID.split('_')[1]
	caption_ID = ID.split('_')[-1]
	
	# Load entity dict for given IDs
	with open('data/entity_IDs/entity_ID_dict_'+image_ID+'_'+caption_ID+'.pickle', 'rb') as handle:
		ent_dict = pickle.load(handle)

	# Load scene graph labels for given image
	with open('data/sg_nodes/sg_categories_'+image_ID+'.pickle', 'rb') as handle:
		sg_categories = pickle.load(handle)

	# Load scene graph nodes for given image
	with open('data/sg_nodes/sg_nodes_'+image_ID+'.pickle', 'rb') as handle:
		sg_nodes = pickle.load(handle)

	# Initialize 
	adjacency_matrix_dict = {}
	no_concept_for_phrase = 0

	# Set phrase concepts
	concepts = concepts.strip()
	phrase_concepts = ast.literal_eval(concepts)

	# Set scene graph concepts
	sg_concepts = sg_categories    # or : 	# sg_nodes = sg_nodes_ID_dict[image_ID]  && list(set([n.category for n in sg_nodes]))

	# Load subgraph (or generate with load_subgraph=False)
	sub_CN = conceptnet.get_subgraphs(ID, cn_graph_full, sg_concepts, phrase_concepts, load_subgraph=True)
	
	# Load ranked phrases 
	phrase_ranks, ranked_sg_by_phrase = conceptnet.rank_concepts_by_relevance(sub_CN, sg_concepts, phrase_concepts)
	
	# Initalize
	phrase_groups_by_ID = {}
	phrase_group_ranked = {}
	group_coordinates = {}
	phrase_names = {}
	phrase_number = {}
	
	# Set scene graph nodes (with ConceptNet entry)
	sg_nodes = [n for n in sg_nodes if sub_CN.has_node(n.category)]

	# Load phrase concepts
	with open('data/phrase_concepts/phrase_concepts_dict_'+image_ID+'_'+caption_ID+'.pickle', "rb") as input_file:
		phrase_concepts = pickle.load(input_file)

	# Initalize
	used_concept_phrases = []
	not_used_concept_phrases = []
	missing = []
	used_keys = []

	# Get phrase groups by entity ID
	for phrase in phrase_concepts:
	
		# Skip if there is no node in subgraph
		if sub_CN.has_node(phrase.concept_text):

			# Append phrase to dict by ID
			if phrase.ent_ID in phrase_groups_by_ID.keys():
				phrase_groups_by_ID[phrase.ent_ID].append(phrase)

			else:
				phrase_groups_by_ID[phrase.ent_ID] = [phrase]
			
			# Save used entity IDs (for tracking)
			used_concept_phrases.append(phrase.ent_ID)

		else:
			# Save unused entity IDs (for tracking)
			not_used_concept_phrases.append(phrase.ent_ID)

	# Get phrase groups (entity IDs)
	phrase_groups = list(phrase_groups_by_ID.keys())

	# Get all entity dict keys (all contained entities)
	all_keys = list(set(list(ent_dict.keys())))

	# Set the used concepts phrase IDs (found and not-found)
	used_concept_phrases = list(set(used_concept_phrases))
	not_used_concept_phrases = list(set(not_used_concept_phrases))

	# Check which entities are missing
	for phrase in not_used_concept_phrases:
		if phrase not in used_concept_phrases:
			missing.append(phrase)
			no_concept_for_phrase += 1

	# Set array for adjacency similarity
	adj_similarity = np.zeros((len(all_keys), len(sg_nodes)))
	
	# For each group (same entity ID)
	for group_idx, group_ID in enumerate(all_keys):
		
		# If group/entity ID is not found, skip
		if group_ID not in phrase_groups_by_ID.keys():
				continue
		else:
			# Get phrases for given group/entity ID
			phrases = phrase_groups_by_ID[group_ID]
			used_keys.append(group_ID)
			
			# Assume that there are phrases to process
			no_phrases = False

			# Only use phrases with entity
			phrases = [p for p in phrases if p != None and p.entity != None]
		
			# If there are phrases
			if phrases:
				
				# Get coordinates and phrase text example 
				coordinates = [p.entity.coordinates for p in phrases][0]
				phrase_name = [p.concept_text for p in phrases][0]
				
				# Save
				group_coordinates[group_ID] = coordinates
				phrase_names[group_ID] = phrase_name

				# Is entity singular/plural? (max vote)
				phrase_is_plural = max([p.is_plural for p in phrases])
				phrase_number[group_ID] = phrase_is_plural

			else:
				no_phrases = True

			# For each scene graph node, get personalized Pagerank
			for sg_idx, sg_concept in enumerate(sg_nodes):

				# Skip if no phrases
				if no_phrases:
					total_ranking = []
					
				else:
					# Check whether concept is plural
					sg_is_plural = sg_concept.label.endswith('pl')
					
					total_ranking = []

					# For each phrase, 
					for phrase in phrases:
						
						# If number comparison on, skip if not same number
						if check_number and phrase_is_plural != sg_is_plural:
							continue

						# Get phrase concept text
						phrase = phrase.concept_text

						# For each phrase, save personalized PageRank
						if phrase in ranked_sg_by_phrase.keys():
							sg_ranking = {k:v for k,v in ranked_sg_by_phrase[phrase]}
							total_ranking.append(sg_ranking[sg_concept.category])

				# Get mean of all pageranks (if there are any)
				if total_ranking:
					total_ranking = sum(total_ranking)/len(total_ranking)
				else:
					total_ranking = 0
				
				# Save as array and dict
				adj_similarity[group_idx, sg_idx] = total_ranking
				adjacency_matrix_dict[(group_ID, sg_concept.label)] = total_ranking


	# Save complete adjacency matrix with personalized Pageranks
	with open('data/pageranks_n/pagerank_adj_matrix_'+image_ID+'_'+caption_ID+'.pickle', 'wb') as handle:
	   pickle.dump(adjacency_matrix_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():

	# If Multiprocessing
	if cfg.MULTI_PRO_PAGERANKS:
		
		# Cluster settings
		cluster = SLURMCluster(
			queue='main',
			cores=8,
			memory="36000 MB",
			processes = 7,
			job_extra = ['--job-name=get_pagerank',
						'--mail-type=NONE',
						'-e /home/students/suter/logs_suter/worker_logs/slurm-%j.err',
						'-o /home/students/suter/logs_suter/worker_logs/slurm-%j.out',
						'--time=72:00:00'],
			env_extra = ['PATH=/opt/slurm/bin:$PATH'] )
			
		
		# Cluster client
		cluster.scale(jobs=15)
		client = Client(cluster)


	# Load which files to use
	with open(cfg.FILES_TO_USE) as infile:
		files_to_use = infile.read().split('\n')

	# Initalizing
	adj_matrix_by_caption = {}
	total = 0
	correct = 0
	final_no_concept_for_phrase = 0

	# Loading phrase concepts
	with open('data/phrase_concepts.txt', 'r') as infile:
		data = infile.read()
		phrase_concepts_data = data.split('\n')
		phrase_concepts_data = [c for c in phrase_concepts_data if c.strip() != '']
	
	# Get all files in pageranks dir
	files_in_dir = [f for f in os.listdir('data/pageranks_n/')]
	processed_samples = []

	# Get processed samples
	for file in files_in_dir:
		parts = file.split('_')
		image_ID = parts[-2]
		caption_ID = parts[-1].strip('.pickle')
		processed_samples.append((image_ID, caption_ID))
		
	# Get all files in subgraphs
	existing_subgraphs = [f for f in os.listdir('data/subgraphs/') if "incl" in f and "red" in f]
	subgraph_samples = []

	# Get IDs of existing subgraphs
	for file in existing_subgraphs:
		parts = file.split('_')
		image_ID = parts[1]
		caption_ID = parts[3]
		subgraph_samples.append((image_ID, caption_ID))


	# Initialize
	futures = []
	chunk_counter = 0

	# For each entry in phrase concepts
	for line in phrase_concepts_data:

		# Get IDs
		ID, concepts = line.split('\t')
		image_ID = ID.split('_')[1]
		caption_ID = ID.split('_')[-1]

		# Skip is already processed
		if (image_ID, caption_ID) in processed_samples:
			continue
			
		# Skip if there is no subgraph for given ID
		if (image_ID, caption_ID) not in subgraph_samples:
			continue

		# Skip if file is not to be processed
		if image_ID not in files_to_use:
			continue
			
		# Multiprocessing
		if cfg.MULTI_PRO_PAGERANKS:

			# Set up client
			future = client.submit(get_pagerank, line, check_number=True)
			futures.append(future)
			chunk_counter += 1

			# Process chunks of 50
			if chunk_counter >= 50:
				client.gather(futures)
				futures = []
				chunk_counter = 0
	
			# Gather results
			client.gather(futures)
			
		# Without multiprocessing
		else:
			get_pagerank(line)


if __name__ == "__main__":
	main()
