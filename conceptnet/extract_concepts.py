# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# extract_concepts.py
# ----------------
# Extract phrase concepts with entity IDs for image/caption pairs


# Import Statements
import pickle
import sys
import re
import os

import Scene_Graph as sg
import concept_config as cfg

import spacy
nlp = spacy.load("en_core_web_sm")

import networkx as nx

from dask.distributed import Client
from dask_jobqueue import SLURMCluster


def get_ent_ID(words_index, text_tags, pos_tags, image_ID, caption_ID, dataset):
	"""Get the entity ID and entity for given words and indices."""

	# Initalize entity ID list
	ent_ID = []

	# Try to look up entity ID by word and index
	try:
		# Get all possbile keys (index, word)
		noun_words = [(words_index[text_tags[i]], text_tags[i]) for i in range(len(text_tags)) if pos_tags[i] == 'NOUN']

		# Get all entity IDs that can be retrieved
		ent_ID = [dataset.word_ent_ID_dict[(image_ID, caption_ID)][(id, word)] for (id,word) in noun_words]

	# Otherwise
	except KeyError:

		# Get all keys from word entity dict
		keys = dataset.word_ent_ID_dict[(image_ID, caption_ID)].keys()

		# For each index
		for i in range(len(text_tags)):

			# Get word
			word = text_tags[i]

			try:
				# Find key with fitting word
				key = next((i,w) for (i,w) in keys if w == word)

				# Save entity ID
				ent_ID.append(dataset.word_ent_ID_dict[(image_ID, caption_ID)][key])

			# Otherwise, append None
			except StopIteration:
				ent_ID.append(None)

	# If exactly one possible unique entity ID, use that
	if len(list(set(ent_ID))) == 1:
		ent_ID = ent_ID[0]

	else:
		# Get all entity IDs that are not None
		ent_ID = [ID for ID in ent_ID if ID is not None]

		# If exactly one possible uniquee entity ID, use that
		if len(list(set(ent_ID))) == 1:
			ent_ID = ent_ID[0]

		# If more than one, discard word as phrase concept (connected to several entities)
		else:
			ent_ID = None

	# Get entity by entity ID
	if ent_ID is None:
		entity = None
	else:
		entity = dataset.ent_ID_dict[(image_ID, caption_ID)][ent_ID]

	# Return results
	return ent_ID, entity


def get_phrase_concepts(line, dataset, BBOXES_DIR):
	"""Extract phrase concepts for given caption."""

	# Get ID and sent
	split_line = line.split('\t')
	ID = split_line[0]
	sent = split_line[1]

	# Get image and caption ID
	split_ID = ID.split('#')
	image_ID = split_ID[0].strip('.jpg')
	caption_ID = split_ID[1]

	# Load ConceptNet graph
	cn_graph = nx.read_gpickle(cfg.CONCEPTNET_DIR)
	cn_nodes = list(cn_graph.nodes)

	# Initialize
	concept_phrase_ID_dict = {}
	phrase_concepts = []
	possible_concepts = []

	# Parse caption
	parsed_phrase = nlp(sent)  # token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_

	# Load Scene Graph
	scene_graph = sg.SceneGraph(image_ID, sg_data_dir=cfg.BBOXES_DIR)
	scene_graph.get_nodes()

	# Get categories
	category_dict = scene_graph.get_label_dict()
	categories = [c for c in category_dict.keys() if not c.endswith('_')]

	# Save scene graph nodes
	with open('data/sg_nodes/sg_nodes_'+image_ID+'.pickle', 'wb') as handle:
	   pickle.dump(scene_graph.nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Save scene graph categories
	with open('data/sg_nodes/sg_categories_'+image_ID+'.pickle', 'wb') as handle:
	   pickle.dump(categories, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Save entity ID dict
	with open('data/entity_IDs/entity_ID_dict_'+image_ID+'_'+caption_ID+'.pickle', 'wb') as handle:
	   pickle.dump(dataset.ent_ID_dict[(image_ID, caption_ID)], handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Clean caption sents (mainly space replacements)
	sent = re.sub(r'\s\((\w+)\)\s',' ( \g<1> ) ', sent)
	sent = re.sub(r'\s\((\w+)\s',' ( \g<1> ', sent)
	sent = re.sub(r'\s(\w+)\)\s',' \g<1> ) ', sent)

	# Index and words
	words_index = {w:i for i,w in enumerate(sent.split(' '))}

	# Get the nouns and lemmas
	nouns_index = [(i,w.text) for i,w in enumerate(parsed_phrase) if w.pos_ == 'NOUN']
	lemmas_index = [(i,w.lemma_) for i,w in enumerate(parsed_phrase) if w.pos_ == 'NOUN']

	# Get all nouns from caption (lemmatized and text form)
	nouns = [n for (i,n) in nouns_index+lemmas_index]

	# For each found noun, get phrase
	for i, (index, word) in enumerate(nouns_index):

		# Default
		ent_ID = None
		entity = None

		# Get lemma by index
		lemma = lemmas_index[i][1]

		# Get index of noun/lemma
		if word in words_index.keys():
			index = words_index[word]

			# Retrieve entity ID (if possilble)
			try:
				ent_ID = dataset.word_ent_ID_dict[(image_ID, caption_ID)][(index,word)]
			except KeyError:
				ent_ID = None

		# If no entity ID is found so far
		if ent_ID is not None:

			# Get keys for word entity dict
			keys = dataset.word_ent_ID_dict[(image_ID, caption_ID)].keys()
			try:
				# Try to find suitable entry
				key = next((i,w) for (i,w) in keys if w == word)
				ent_ID = dataset.word_ent_ID_dict[(image_ID, caption_ID)][key]

			# Otherwise set to None
			except StopIteration:
				ent_ID = None

		# Set entity based on entity ID
		if ent_ID is not None:
			entity = dataset.ent_ID_dict[(image_ID, caption_ID)][ent_ID]

		# Get POS tags for noun
		noun_pos_tag = [w.tag_ for i,w in enumerate(parsed_phrase) if i == index and w.text == word]

		# Check whether plural or not
		if 'NNS' in noun_pos_tag:
			is_plural = True
		else:
			is_plural = False

		# Save word and lemma with ID
		concept_phrase_ID_dict[word] = ent_ID
		concept_phrase_ID_dict[lemma] = ent_ID

		# If non-empty entity, save phrase as phrase concept
		if entity is not None:
			# Save word form
			phrase_concepts.append(sg.PhraseConcept(line, image_ID, caption_ID, word, ent_ID, entity, is_plural))
			# Save lemma
			phrase_concepts.append(sg.PhraseConcept(line, image_ID, caption_ID, lemma, ent_ID, entity, is_plural))


	# For each bigram, check whether it is suitable phrase
	for word_ind, (w1,w2) in enumerate(zip(parsed_phrase, parsed_phrase[1:])):

		# Get pos tags and word forms
		pos_tags = [w1.pos_, w2.pos_]
		text_tags = [w1.text, w2.text]

		# If noun is in included and does not end in adverb
		if "NOUN" in pos_tags and w2.pos_ != 'ADP':

			# Get pos tags
			noun_pos_tag = [w.tag_ for w in [w1,w2] if w.pos_ == 'NOUN']

			# Check whether plural or not
			if 'NNS' in noun_pos_tag:
				is_plural = True
			else:
				is_plural = False

			# Generate concept (with underscore)
			concept_text = w1.text+'_'+w2.text
			concept_lemma = w1.lemma_+'_'+w2.lemma_

			# Append to possible concepts
			possible_concepts.append(concept_text)
			possible_concepts.append(concept_lemma)

			# Get entity and ID
			ent_ID, entity = get_ent_ID(words_index, text_tags, pos_tags, image_ID, caption_ID, dataset)

			# Save phrase concepts with ID
			concept_phrase_ID_dict[concept_text] = ent_ID
			concept_phrase_ID_dict[concept_lemma] = ent_ID

			# Save phrase concepts as lemma and word form
			if entity is not None:
				phrase_concepts.append(sg.PhraseConcept(line, image_ID, caption_ID, concept_text, ent_ID, entity, is_plural))
				phrase_concepts.append(sg.PhraseConcept(line, image_ID, caption_ID, concept_lemma, ent_ID, entity, is_plural))


	# For each trigram, check whether it is suitable phrase
	for word_ind, (w1, w2, w3) in enumerate(zip(parsed_phrase, parsed_phrase[1:], parsed_phrase[2:])):

		# Get pos tags and word forms
		pos_tags = [w1.pos_, w2.pos_, w3.pos_]
		text_tags = [w1.text, w2.text, w3.text]

		# If noun is in included and does not end in adverb
		if 'NOUN' in pos_tags and w3.pos_ != 'ADP':

			# Get pos tags
			noun_pos_tag = [w.tag_ for w in [w1,w2] if w.pos_ == 'NOUN']

			# Check whether plural or not
			if 'NNS' in noun_pos_tag:
				is_plural = True
			else:
				is_plural = False

			# Generate concept (with underscore)
			concept_text = w1.text+'_'+w2.text+'_'+w3.text
			concept_lemma = w1.lemma_+'_'+w2.lemma_+'_'+w3.lemma_

			# Append to possible concepts
			possible_concepts.append(concept_text)
			possible_concepts.append(concept_lemma)

			# Get entity and ID
			ent_ID, entity = get_ent_ID(words_index, text_tags, pos_tags, image_ID, caption_ID, dataset)

			# Save phrase concepts with ID
			concept_phrase_ID_dict[concept_text] = ent_ID
			concept_phrase_ID_dict[concept_lemma] = ent_ID

			# Save phrase concepts as lemma and word form
			if entity is not None:
				phrase_concepts.append(sg.PhraseConcept(line, image_ID, caption_ID, concept_text, ent_ID, entity, is_plural))
				phrase_concepts.append(sg.PhraseConcept(line, image_ID, caption_ID, concept_lemma, ent_ID, entity, is_plural))


	# For each 4-gram, check whether it is suitable phrase
	for word_ind, (w1, w2, w3, w4) in enumerate(zip(parsed_phrase, parsed_phrase[1:], parsed_phrase[2:], parsed_phrase[3:])):

		# Get pos tags and word forms
		pos_tags = [w1.pos_, w2.pos_, w3.pos_, w4.pos_]
		text_tags = [w1.text, w2.text, w3.text, w4.text]

		# If noun is in included and does not end in adverb
		if 'NOUN' in pos_tags and w4.pos_ != 'ADP':

			# Get pos tags
			noun_pos_tag = [w.tag_ for w in [w1,w2] if w.pos_ == 'NOUN']

			# Check whether plural or not
			if 'NNS' in noun_pos_tag:
				is_plural = True
			else:
				is_plural = False

			# Generate concept (with underscore)
			concept_text = w1.text+'_'+w2.text+'_'+w3.text+'_'+w4.text
			concept_lemma = w1.lemma_+'_'+w2.lemma_+'_'+w3.lemma_+'_'+w4.lemma_

			# Append to possible concepts
			possible_concepts.append(concept_text)
			possible_concepts.append(concept_lemma)

			# Get entity and ID
			ent_ID, entity = get_ent_ID(words_index, text_tags, pos_tags, image_ID, caption_ID, dataset)

			# Save phrase concepts with ID
			concept_phrase_ID_dict[concept_text] = ent_ID
			concept_phrase_ID_dict[concept_lemma] = ent_ID

			# Save phrase concepts as lemma and word form
			if entity is not None:
				phrase_concepts.append(sg.PhraseConcept(line, image_ID, caption_ID, concept_text, ent_ID, entity, is_plural))
				phrase_concepts.append(sg.PhraseConcept(line, image_ID, caption_ID, concept_lemma, ent_ID, entity, is_plural))

	# Get caption phrase dictionary
	caption_phrases = dataset.caption_phrase_dict[(image_ID, caption_ID)]

	# Get concepts that have ConceptNet entry
	filtered_possible_concepts = list(set([con for con in possible_concepts if con in cn_nodes]))
	# Get all nouns that have ConceptNet entry
	nouns = list(set([con for con in nouns if con in cn_nodes]))

	# Get all concepts
	concepts = filtered_possible_concepts+nouns

	# For each caption phrase with no concept, try to add entry
	for cap_phrase, id in caption_phrases:

		# If no concpet but entry in ConceptNet
		if cap_phrase not in concepts and cap_phrase in cn_nodes:

			# Look up entity (if found) and save new concept phrase
			if id in dataset.ent_ID_dict[(image_ID, caption_ID)]:
				entity = dataset.ent_ID_dict[(image_ID, caption_ID)][id]
				new_phrase = sg.PhraseConcept('', image_ID, caption_ID, cap_phrase, id, entity, False)
			else:
				new_phrase = sg.PhraseConcept('', image_ID, caption_ID, cap_phrase, id, None, False)

			# Save phrase concepts
			phrase_concepts.append(new_phrase)
			concepts.append(cap_phrase)

	# Save phrase concepts dict
	with open('data/phrase_concepts/phrase_concepts_dict_'+image_ID+'_'+caption_ID+'.pickle', 'wb') as handle:
	   pickle.dump(phrase_concepts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Save concepts as string
	concepts = ["'"+c+"'" for c in concepts]
	concepts_string = "["+",".join(concepts)+"]"

	# Write out phrase concepts
	with open('data/phrase_concepts/phrase_concepts_'+image_ID+'_'+caption_ID+'.txt', 'a') as outfile:

		outfile.write('Image_'+image_ID+'_Caption_'+caption_ID)
		outfile.write('\t')
		outfile.write(concepts_string)
		outfile.write('\n')


def main():

	# If multiprocessing, set cluster and job settings
	if cfg.MULTI_PHRASE_CNCPTS:

		cluster = SLURMCluster(
			queue='main',
			cores=8,
			memory="36000 MB",
			processes = 7,
			job_extra = ['--job-name=extract_concepts',
						'--mail-type=NONE',
						'-e /home/students/suter/logs_suter/worker_logs/slurm-%j.err',
						'-o /home/students/suter/logs_suter/worker_logs/slurm-%j.out',
						'--time=24:00:00'],

			env_extra = ['PATH=/opt/slurm/bin:$PATH'])


	# Load files to process
	with open(cfg.FILES_TO_USE) as infile:
		files_to_use = infile.read().split('\n')

	# Load captions
	with open(cfg.DATA_DIR+"flickr30k-captions/results_20130124.token", "r") as infile:
		data = infile.read()
		caption_data = data.split('\n')

	# Load files that are already processed
	processed_files_dir = [f for f in os.listdir('data/phrase_concepts/') if f.endswith('.pickle')]
	processed_samples = []

	# Save files that are already processed
	for file in processed_files_dir:
		parts = file.split('_')
		image_ID = parts[-2]
		caption_ID = parts[-1].strip('.pickle')
		processed_samples.append((image_ID, caption_ID))
		
	print(len(processed_samples))

	# Load dataset and get captions and entities
	dataset = sg.Dataset(cfg.DATA_DIR)
	dataset.get_captions()
	dataset.get_entities()

	# Cluster settigs
	if cfg.MULTI_PHRASE_CNCPTS:
		cluster.scale(jobs=15)
		print(cluster.job_script())
		client = Client(cluster)

	# Initalize
	futures = []
	chunk_counter = 0

	# For each sample
	for line in caption_data:

		# Skip if empty
		if line.strip() == '':
			continue

	    # Get ID and sent
		split_line = line.split('\t')
		ID = split_line[0]
		sent = split_line[1]

		# Get image and caption ID
		split_ID = ID.split('#')
		image_ID = split_ID[0].strip('.jpg')
		caption_ID = split_ID[1]
		
		print(image_ID, caption_ID)

		# Skip if already processed
		if (image_ID, caption_ID) in processed_samples:
			continue

		# Skip if not in "files to process"
		if image_ID not in files_to_use:
			continue
			
		# Multiprocessing
		if cfg.MULTI_PHRASE_CNCPTS:

			# Count multi-processing chunks
			chunk_counter += 1

			# Run: get phrase concepts for given sample
			future = client.submit(get_phrase_concepts, line, dataset, cfg.BBOXES_DIR)
			futures.append(future)

			# Gather after 50 samples
			if chunk_counter >= 20:
				client.gather(futures)
				futures = []
				chunk_counter = 0

		# Without multi-processing
		else:
			get_phrase_concepts(line, dataset, cfg.BBOXES_DIR)

	# Final gathering of results
	if cfg.MULTI_PHRASE_CNCPTS:
		client.gather(futures)


	# Load all phrase concept files
	phrase_concepts_files = [f for f in os.listdir('data/phrase_concepts/') if f.endswith('.txt')]

	# Write it all out to one file
	with open('data/phrase_concepts.txt', 'w') as outfile:
		outfile.write('')

		# For each file, load concent and copy to new file
		for file in phrase_concepts_files:

			# Load content
			with open('data/phrase_concepts/'+file, 'r') as infile:
				line = infile.read()
				line = line.split('\n')[0]

				# Write out to new file
				with open('data/phrase_concepts.txt', 'a') as outfile:
					outfile.write(line)
					outfile.write('\n')


if __name__ == "__main__":
    main()
