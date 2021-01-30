# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# get_alternatives.py
# ----------------
# Retrieve support concepts (alternatives) and relations 
# for given scene graph and phrase concepts.


# Import statements
import ast
import pickle
import os

import networkx as nx

import conceptnet
import concept_config as cfg

from dask.distributed import Client
from dask_jobqueue import SLURMCluster


def save_alternatives(line):

	# Get IDs
	ID, concepts = line.split('\t')
	image_ID = ID.split('_')[1]
	caption_ID = ID.split('_')[-1]

	# Load scene graph labels for given image
	with open('data/sg_nodes/sg_categories_'+image_ID+'.pickle', 'rb') as handle:
		sg_categories = pickle.load(handle)

	# Load direction dictionary
	with open('data/ConceptNet/dir_dict.pickle', 'rb') as handle:
		dir_dict = pickle.load(handle)

	# Load ConceptNet graph
	cn_graph_full = nx.read_gpickle('data/ConceptNet/conceptnet_full_di_rel_red.gpickle')

	# Load phrase concepts
	concepts = concepts.strip()
	phrase_concepts = ast.literal_eval(concepts)

	# Load scene graph concepts
	sg_concepts = sg_categories

	# Get ConceptNet subgraph
	sub_CN = conceptnet.get_subgraphs(ID, cn_graph_full, sg_concepts, phrase_concepts, load_subgraph=True)

	# Get ranked paths by phrase
	phrase_ranks, ranked_sg_by_phrase = conceptnet.rank_concepts_by_relevance(sub_CN, sg_concepts, phrase_concepts)
	
	# Get shortest paths between relevant pairs
	paths_by_phrase, _ = conceptnet.get_paths_between_relevant_concept_pairs(sub_CN, phrase_concepts, ranked_sg_by_phrase, n_heighest_ranked=10)
  
	# Load phrase concepts
	with open('data/phrase_concepts/phrase_concepts_dict_'+image_ID+'_'+caption_ID+'.pickle', "rb") as input_file:
		phrase_concepts = pickle.load(input_file)

	# Get scene graph and phrase concep relations for alternatives
	sg_pair_rels, phrase_pair_rels = conceptnet.get_alternatives(image_ID, caption_ID, paths_by_phrase, phrase_concepts, dir_dict, sub_CN)

	# Save alternative scene graph concepts
	with open('data/alternatives/alternatives_sg_'+image_ID+'_'+caption_ID+'.pickle', 'wb') as handle:
	   pickle.dump(sg_pair_rels, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Save alternative phrase concepts
	with open('data/alternatives/alternatives_phrase_'+image_ID+'_'+caption_ID+'.pickle', 'wb') as handle:
	   pickle.dump(phrase_pair_rels, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():

	# If multiprocessing
	if cfg.MULTI_PRO_ALTS:
		
		# Cluster settings
		cluster = SLURMCluster(
		queue='main',
		cores=8,
		memory="36000 MB",
		processes = 7,
		job_extra = ['--job-name=get_alternative',
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

	# Load phrase concepts data
	with open('data/phrase_concepts.txt', 'r') as infile:
		data = infile.read()
		phrase_concepts_data = data.split('\n')
		phrase_concepts_data = [c for c in phrase_concepts_data if c.strip() != '']
		
	# Get all files in pageranks dir
	files_in_dir = [f for f in os.listdir('data/alternatives/') if f.startswith('alternatives_sg')]
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
		if cfg.MULTI_PRO_ALTS:

			# Set up client
			future = client.submit(save_alternatives, line)
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
			save_alternatives(line)


if __name__ == "__main__":
	main()
