# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# support_concept_evaluation.py
# ----------------
# Insert support concepts (alternatives) for scene graph objects and 
# phrases and evaluate


# Imports
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from dask.distributed import Client
from dask_jobqueue import SLURMCluster

# Own modules
import embeddings as emb
import box_geometry as box

# Settings
new_file_tag = '_new_3'
ONLY_GOOD_RELATIONS = False
THRESHOLD = 0.5


# Load files
with open('conceptnet_data'+new_file_tag+'/useful_sg_relations.pickle', 'rb') as handle:
   useful_sg_relations = pickle.load(handle)

with open('conceptnet_data'+new_file_tag+'/useful_phrase_relations.pickle', 'rb') as handle:
   useful_phrase_relations = pickle.load(handle)

with open('conceptnet_data'+new_file_tag+'/phrase_relations_acc_dict.pickle', 'rb') as handle:
   phrase_relations_acc_dict = pickle.load(handle)

with open('conceptnet_data'+new_file_tag+'/sg_relations_acc_dict.pickle', 'rb') as handle:
   sg_relations_acc_dict = pickle.load(handle)


# How many useful relations
print('Useful SG relations', len(useful_sg_relations))
print('Useful Phrase relations', len(useful_phrase_relations))


# Get useful relations with acc >= 0.75 and at least 25 instances 
useful_sg_relations = [rel for rel in sg_relations_acc_dict.keys() if sg_relations_acc_dict[rel][0] >= 0.75 and sg_relations_acc_dict[rel][1] >25]
useful_phrase_relations = [rel for rel in phrase_relations_acc_dict.keys() if phrase_relations_acc_dict[rel][0] >= 0.75 and phrase_relations_acc_dict[rel][1] > 25]

# How many useful relations after filtering
print('Useful SG relations', len(useful_sg_relations))
print('Useful Phrase relations', len(useful_phrase_relations))



class Metrics(object):
	"""Metrics class for support concept evaluation."""
	def __init__(self):
		
		# Dictionaries
		self.sg_evaluation_dict = defaultdict(int)
		self.sg_evaluation_dict_align = defaultdict(int)
		self.sg_total_dict = defaultdict(int)

		self.phrase_evaluation_dict = defaultdict(int)
		self.phrase_evaluation_dict_align = defaultdict(int)
		self.phrase_total_dict = defaultdict(int)

		self.default_alignment_accuracies = []
		self.new_alignment_accuracies = []
		self.sg_alignment_accuracy =  defaultdict(list)
		self.phrase_alignment_accuracy =  defaultdict(list)
		
		# Counters
		self.sg_better_score = 0
		self.sg_total =  0

		self.phrase_better_score = 0
		self.phrase_total = 0

		self.sg_alignment_success = 0
		self.sg_alignment_total = 0

		self.phrase_alignment_success = 0
		self.phrase_alignment_total = 0

		self.sg_better_alignment_score = 0
		self.phrase_better_alignment_score = 0

		self.sg_same_alignment_score = 0
		self.phrase_same_alignment_score = 0

		self.sg_better_alignment_total = 0
		self.phrase_better_alignment_total = 0

		self.sg_combined_better_alignment = 0
		self.phrase_combined_better_alignment = 0

		self.sg_combined_same_alignment = 0
		self.phrase_combined_same_alignment = 0

		self.sg_combined_total_alignment = 0
		self.phrase_combined_total_alignment = 0

def change_label(original_sg_objects, old_label, new_label, new_emb_dict=None):
	"""Change old label to new label."""

	# Initialize
	new_labels = []
	new_embeddings = []

	sg_objects = [[],[],[]]

	# For each label in the original objects
	for i, label in enumerate(original_sg_objects[0]):
		
		# Get category and ID
		cat, ID = label.split('-')

		# If cat is the old label that is to be replaced
		if cat == old_label:
			
			# Save new label
			new_labels.append(new_label+'-'+ID)

			# If new embeddings dict, combine old and new label embedding
			if new_emb_dict:
				
				# Get new object embedding (saved merged embeddings) -- not used
				obj_emb = new_emb_dict[old_label]
				
				# Get combined embedding for old and new label
				combo_emb_array = np.zeros((2,300))
				combo_emb_array[0,:] = emb.embed(old_label)
				combo_emb_array[1,:] = emb.embed(new_label)
				combo_emb_array = combo_emb_array.mean(axis=0)
				obj_emb = combo_emb_array
				
			# Ohterwise only new embedding
			else:
				obj_emb = emb.embed(new_label)
				
			# Save new embedding
			new_embeddings.append(obj_emb)

		# If not label to be changed, save original data
		else:
			new_labels.append(label)
			new_embeddings.append(original_sg_objects[1][i])
	
	# Get new labels, embeddings and coordinates
	sg_objects[0] = new_labels
	sg_objects[1] = new_embeddings
	sg_objects[2] = original_sg_objects[2]

	return sg_objects
	

def get_adj_matrix(filename, old_label=None, new_label=None, new_phrase=None, old_phrase=None, new_emb_dict=None, new_phrase_emb_dict=None):
	"""Get the adjacency matrix."""

	# Load scene graph objects and phrase embeddings
	npzfile = np.load(filename, allow_pickle=True)
	sg_objects = npzfile['name1'].tolist()
	phrase_emb = npzfile['name2']
	disc_indices_array = npzfile['name3'].tolist()
	match_array = npzfile['name4']

	# Change the old label to new label
	if old_label and new_label:
		sg_objects = change_label(sg_objects, old_label, new_label, new_emb_dict)

	# If a new phrase is used for comparing
	if new_phrase:
		
		# If new phrase emb dict is used
		if new_phrase_emb_dict:
			
			# Get new phrase embedding (saved merged embeddings) -- not used
			phrase_emb = new_phrase_emb_dict[old_phrase]
			
			# Get combination of old and new phrase
			combo_emb_array = np.zeros((2,300))
			combo_emb_array[0,:] = emb.embed(old_phrase)
			combo_emb_array[1,:] = emb.embed(new_phrase)
			combo_emb_array = combo_emb_array.mean(axis=0)
			phrase_emb = combo_emb_array

		# Otherwise just embed new phrase
		else:
			phrase_emb = emb.embed(new_phrase)

	# Get all embeddings
	embeddings = np.array(sg_objects[1])

	# Get cosine similarity
	cosim = 1 - cosine_similarity(
		phrase_emb.reshape(1, -1), embeddings)[0]

	# Convert everything close 0 to 0.0
	cosim[np.isclose(0.0, cosim)] = 0.0

	# Change cosine simiarlity to 1.0 if object is to be discarded
	for ind in disc_indices_array:
		cosim[ind] = 1.0

	return cosim, match_array


def get_alignment(adj_matrix, phrase_indices, sg_object_array, ent_IDs):
	"""Get alignment score given adjacency matrix, phrases and objects."""
	
	# Initalize
	success = 0
	total = 0
	used_labels = []

	# Get linear sim assignment
	tag_IDX, object_IDX = linear_sum_assignment(adj_matrix)

	# For each pairing
	for tag_ind, object_idx in zip(tag_IDX, object_IDX):
		
		# Get ideal object label and cat
		tag_idx = phrase_indices[tag_ind]
		objects_tag = sg_object_array[0][object_idx]
		indices = [object_idx]
		label = sg_object_array[0][indices[0]]
		cat = label.split('-')[0]

		# If not plural, get biggest (unused) bounding box with given label
		if not label.endswith('pl'):
			
			# Get biggest unused box with this label
			best_solution = sorted([(i, label) for i,label in enumerate(sg_object_array[0]) if label.split('-')[0] == cat if label not in used_labels])[0]
			best_index, best_label = best_solution
			indices = [best_index]
			used_labels.append(best_label)

		# Get first match
		index = indices[0]

		# Get coordinates for phrase scene graph object
		phrase_coord = ent_IDs[tag_idx].coordinates
		sg_coord = sg_object_array[2][index]

		# Check IoU score
		if box.overlap_iou(phrase_coord, sg_coord):
			success += 1
		total += 1

	# Get alignment accuracy
	alignment_acc = success/total*100
	return alignment_acc

def get_sg_alternatives(image_ID, caption_ID, phrase_indices,
						sg_object_array, res_matrix, default_adj_matrix,\
						default_score, default_alignment_acc, ONLY_GOOD_RELATIONS, \
						ent_IDs, emb_version=False):
	"""Get scene graph object alternatives and evaluate on phrase grounding task."""

	# Initialize
	better_scores = 0
	total_scores = 0
	successful_relations = []
	successful_relations_align = []
	total_relations = []
	sg_alignments_per_rel = []
	alignment_accuracies = []

	better_alignments = 0
	same_alignments = 0
	total_alignments = 0

	combined_better_alignment = 0
	combined_same_alignment = 0

	# Load alternatives and relation accuracies (success rate)
	with open("../ConceptNet/data/alternatives_red/alternatives_sg_"+image_ID+"_"+caption_ID+".pickle", "rb") as input_file:
		alternative_sg_objs = pickle.load(input_file)

	if ONLY_GOOD_RELATIONS:
		with open('conceptnet_data/useful_sg_relations.pickle', 'rb') as handle:
		   useful_sg_relations = pickle.load(handle)

		with open('conceptnet_data/sg_relations_acc_dict.pickle', 'rb') as handle:
		   sg_relations_acc_dict = pickle.load(handle)

		# Filter for alternatives with relation accuracy (sucess rate) greather than 0.75 and more than 25 instances
		useful_sg_relations = [rel for rel in sg_relations_acc_dict.keys() if sg_relations_acc_dict[rel][0] >= 0.75 and sg_relations_acc_dict[rel][1] > 25]
		alternative_sg_objs = [(_, old_label, new_label, relation, _, path_length) for (_, old_label, new_label, relation, _, path_length) in alternative_sg_objs if tuple(relation) in useful_sg_relations]


	# Initalize
	new_label_dict = {}
	new_combo_label_emb = {}
	old_label_index = {}

	# Create new label dictionary
	for _, old_label, _, _, _, _ in alternative_sg_objs:
		new_label_dict[old_label] = []
	for _, old_label, new_label, _, _, _ in alternative_sg_objs:
		new_label_dict[old_label].append(new_label)

	# Get combination embedding for new labels
	for ol_idx, old_label in enumerate(new_label_dict.keys()):
		
		# Get new labels
		old_label_index[old_label] = ol_idx
		new_labels = new_label_dict[old_label]
		
		# Create array
		new_shape = [len(new_labels)+1]+list(emb.embed(new_labels[0]).shape)
		combo_emb_array = np.zeros(tuple(new_shape))
		
		# Use old label as first embedding
		combo_emb_array[0,:] = emb.embed(old_label)
		
		# Add new labels to embedding array
		for label_idx, new_label in enumerate(new_labels):
			combo_emb_array[label_idx+1,:] = emb.embed(new_label)

		# Get mean and save combo embedding
		combo_emb_array = combo_emb_array.mean(axis=0)
		new_combo_label_emb[old_label] = combo_emb_array

	# Create combo matrix array
	combo_matrix =  np.zeros((len(alternative_sg_objs)+1, len(phrase_indices), len(sg_object_array[0])))

	# Fill with default
	combo_matrix[0,:,:] = default_adj_matrix
	
	processed_ones = []
	
	# Iterate over every alternative object (label)
	for alt_sg_id, alt_sg in enumerate(alternative_sg_objs):

		# Get elements
		_, old_label, new_label, relation, _, path_length = alt_sg

		# Create adj matrix array
		adj_matrix = np.zeros((len(phrase_indices), len(sg_object_array[0])))

		# Discard relations that are not used
		if ONLY_GOOD_RELATIONS:
			if tuple(relation) not in useful_sg_relations:
				continue

		# Iterate over each phrase 
		for i, phrase_id in enumerate(phrase_indices):

			# Get filename for loading data
			filename = './.cosim_collection/'+image_ID+'_'+caption_ID+'_'+phrase_id+'.npz'

			# Get cosine similarity
			if emb_version:
				cosim, _ = get_adj_matrix(filename, old_label, new_label, new_emb_dict=new_combo_label_emb)
			else:
				cosim, _ = get_adj_matrix(filename, old_label, new_label)

			# Save cosine similarity for given phrase/object pair
			adj_matrix[i,:] = cosim

		# Save all cosine similarities for all phrases
		combo_matrix[old_label_index[old_label]+1,:,:] = adj_matrix
	
		# Get weighted sum score
		score = np.sum(np.multiply(adj_matrix, res_matrix))

		# If score is smaller than default score (better!), consider success
		# Optional: set minimum difference
		if score < default_score: # and ((score-default_score) >=0.1):
			better_scores += 1
			successful_relations.append(tuple(relation))
		
		# Track and count relations
		total_relations.append(tuple(relation))
		total_scores += 1
	
		# Track accuracy scores
		alignment_acc = get_alignment(adj_matrix, phrase_indices, sg_object_array, ent_IDs)
		sg_alignments_per_rel.append((tuple(relation),alignment_acc))
		alignment_accuracies.append(alignment_acc)

		# Check whether better/worse/same alignment
		if alignment_acc > default_alignment_acc:
			better_alignments += 1
		if alignment_acc == default_alignment_acc:
			same_alignments += 1
		total_alignments += 1

		# Determine better alignment sucesses
		if alignment_acc >= default_alignment_acc:
			successful_relations_align.append(tuple(relation))

	# Get mean of cosine similarity matrices
	adj_matrix = combo_matrix.mean(axis=0)

	# Get phrase grounding accuracy 
	combo_alignment_acc = get_alignment(adj_matrix, phrase_indices, sg_object_array, ent_IDs)

	# Count improvements/non-improvements
	if combo_alignment_acc > default_alignment_acc:
		combined_better_alignment += 1
	elif combo_alignment_acc == default_alignment_acc:
		combined_same_alignment += 1

	return better_scores, total_scores, successful_relations, total_relations, \
	sg_alignments_per_rel, alignment_accuracies, better_alignments, same_alignments, total_alignments, \
	combo_alignment_acc, combined_better_alignment, combined_same_alignment, successful_relations_align
	

def get_phrase_alternatives(image_ID, caption_ID, phrase_indices, sg_object_array,\
							res_matrix, default_adj_matrix, default_score,\
							default_alignment_acc, ONLY_GOOD_RELATIONS, ent_IDs,\
							emb_version=False):								
	"""Get alternative phrases and evaluate on phrase grounding task."""

	# Initialize
	better_scores = 0
	total_scores = 0
	successful_relations = []
	successful_relations_align = []
	total_relations = []
	phrase_alignments_per_rel = []
	alignment_accuracies = []

	better_alignments = 0
	same_alignments = 0
	total_alignments = 0

	combined_better_alignment = 0
	combined_same_alignment = 0

	# Load altenrative phrases
	with open("../ConceptNet/data/alternatives_red/alternatives_phrase_"+image_ID+"_"+caption_ID+".pickle", "rb") as input_file:
		alternative_phrases = pickle.load(input_file)

	# Load specific alternatives (based on relation)
	if ONLY_GOOD_RELATIONS:
		
		# Load relations and sucess rates
		with open('conceptnet_data/useful_phrase_relations.pickle', 'rb') as handle:
		   useful_phrase_relations = pickle.load(handle)
		with open('conceptnet_data/phrase_relations_acc_dict.pickle', 'rb') as handle:
		   phrase_relations_acc_dict = pickle.load(handle)

		# Get useful phrase relations (minimal success rate and occurences)
		useful_phrase_relations = [rel for rel in phrase_relations_acc_dict.keys() if phrase_relations_acc_dict[rel][0] >= 0.75 and phrase_relations_acc_dict[rel][1] > 25]
	
		# Get alternative phrases based on acceptable relations 
		alternative_phrases = [(_, old_label, new_label, relation, _, path_length) for (_, old_label, new_label, relation, _, path_length) in alternative_phrases if tuple(relation) in useful_phrase_relations]

	# Initialize
	new_phrase_dict = {}	
	new_combo_phrase_emb = {}
	old_phrase_index = {}


	# Get dictionary for new phrases
	for _, old_label, _, _, _, _ in alternative_phrases:
		new_phrase_dict[old_label] = []
	for _, old_label, new_label, _, _, _ in alternative_phrases:
		new_phrase_dict[old_label].append(new_label)

	# Get combo embeddings array for phrases
	for ol_idx,old_label in enumerate(new_phrase_dict.keys()):
		
		# Get new phrases
		new_labels = new_phrase_dict[old_label]
		old_phrase_index[old_label] = ol_idx

		# Create array for combined embeddings for phrases
		new_shape = [len(new_labels)+1]+list(emb.embed(new_labels[0]).shape)
		combo_emb_array = np.zeros(tuple(new_shape))
		
		# Use original phrase as first embedding
		combo_emb_array[0,:] = emb.embed(old_label)
		
		# Add embeddings for new phrases
		for label_idx, new_label in enumerate(new_labels):
			combo_emb_array[label_idx+1,:] = emb.embed(new_label)
	
		# Get mean over embeddings and save
		combo_emb_array = combo_emb_array.mean(axis=0)
		new_combo_phrase_emb[old_label] = combo_emb_array


	# Create combo matrix and fill in original 
	combo_matrix = np.zeros((len(alternative_phrases)+1, len(phrase_indices), len(sg_object_array[0])))
	combo_matrix[0,:,:] = default_adj_matrix

	processed_ones = []

	# Iterate over phrases
	for alt_phrase_id, alt_phrase in enumerate(alternative_phrases):

		# Create array
		adj_matrix = np.zeros((len(phrase_indices), len(sg_object_array[0])))

		# Get new phrases
		new_phrase_ID, old_phrase, new_phrase, relation, _, path_length = alt_phrase

		# Discard disqualified relations
		if ONLY_GOOD_RELATIONS:
			if tuple(relation) not in useful_phrase_relations:
				continue

		# Iterate over phrase set
		for i, phrase_id in enumerate(phrase_indices):

			# Generate filename
			filename = './.cosim_collection/'+image_ID+'_'+caption_ID+'_'+phrase_id+'.npz'

			# If phrase is phrase to be replaced
			if new_phrase_ID == phrase_id:
				
				# Get cosine similarity (either with combined embeddings or single)
				if emb_version:
					cosim, _ = get_adj_matrix(filename, new_phrase=new_phrase, old_phrase=old_phrase, new_phrase_emb_dict=new_combo_phrase_emb)
				else:
					cosim, _ = get_adj_matrix(filename, new_phrase=new_phrase)
			
			# Get cosine similarity without change
			else:
				cosim, _ = get_adj_matrix(filename)
			
			# Save cosine similarity
			adj_matrix[i,:] = cosim

	
		# Save cosine similarities as adj matrix for all phrases
		combo_matrix[old_phrase_index[old_phrase]+1,:,:] = adj_matrix

		# Get weighted sum score
		score = np.sum(np.multiply(adj_matrix, res_matrix))
		
		# If score is smaller than default score (better!), consider success
		# Optional: set minimum difference	
		if score < default_score:# and ((score-default_score) >=0.1):
			better_scores += 1
			successful_relations.append(tuple(relation))

		# Track and count relations
		total_relations.append(tuple(relation))
		total_scores += 1
		
		# Get phrase grounding accuracy
		alignment_acc = get_alignment(adj_matrix, phrase_indices, sg_object_array, ent_IDs)

		# Check whether better/worse/same alignment
		if alignment_acc > default_alignment_acc:
			better_alignments += 1
		if alignment_acc == default_alignment_acc:
			same_alignments += 1
		total_alignments += 1
	
		# Determine whether replacement is successful
		if alignment_acc >= default_alignment_acc:
			successful_relations_align.append(tuple(relation))

		# Save relations and accuracies
		phrase_alignment_per_rel = (tuple(relation),alignment_acc)
		alignment_accuracies.append(alignment_acc)

	# Get mean of cosine similarities
	adj_matrix = combo_matrix.mean(axis=0)

	# Get accuracy for combined version
	combo_alignment_acc = get_alignment(adj_matrix, phrase_indices, sg_object_array, ent_IDs)

	# Count improvements/non-improvements
	if combo_alignment_acc > default_alignment_acc:
		combined_better_alignment += 1
	elif combo_alignment_acc == default_alignment_acc:
		combined_same_alignment += 1

	return better_scores, total_scores, successful_relations, total_relations, \
	phrase_alignments_per_rel, alignment_accuracies, better_alignments, same_alignments, total_alignments, \
	combo_alignment_acc, combined_better_alignment, combined_same_alignment, successful_relations_align
	

def get_default_scores(files, image_ID, caption_ID, phrase_indices, ent_IDs):
	"""Get default/original scores and adjacency matrix for data without replacements/alternatives."""

	# Generate filename
	filename = './.cosim_collection/'+image_ID+'_'+caption_ID+'_'+phrase_indices[0]+'.npz'

	# Load data
	npzfile = np.load(filename, allow_pickle=True)
	sg_object_array = npzfile['name1'].tolist()

	# Create arrays
	default_adj_matrix = np.zeros((len(phrase_indices), len(sg_object_array[0])))
	res_matrix = np.zeros((len(phrase_indices), len(sg_object_array[0])))
	
	# For each phrase, get result matrix and cosine similarity
	for i, phrase_id in enumerate(phrase_indices):
	
		# Generat filename 
		filename = './.cosim_collection/'+image_ID+'_'+caption_ID+'_'+phrase_id+'.npz'
		
		# Get cosine similarity and result array
		cosim, match_array = get_adj_matrix(filename)

		# Get mask for 0 instances and change to desired value
		array_mask = np.where(match_array == 0)
		match_array[array_mask] = 0.0  # or -0.5

		# Save cosine similarity and result matrix
		default_adj_matrix[i,:] = cosim
		res_matrix[i,:] = match_array

	# Get default/original score (weighted sum)
	default_score = np.sum(np.multiply(default_adj_matrix, res_matrix))

	# Get phase grounding alignment
	default_alignment_acc = get_alignment(default_adj_matrix, phrase_indices, sg_object_array, ent_IDs)

	return sg_object_array, res_matrix, default_adj_matrix, default_score, default_alignment_acc


def process_results(results, metrics):
	"""Process the results of the alternative evaluation; save and count accordingly. """
	
	# Iterate over saved results
	for result in results:

		# Continue if input is empty
		if result == None:
			continue

		# Get various results
		sg_results, phrase_results, default_alignment_acc = result
		
		# Get result parts for scene graph object replacement results
		better_scores, total_scores = sg_results[0], sg_results[1]
		successful_relations, total_relations, successful_relations_align = sg_results[2], sg_results[3], sg_results[-1]
		sg_alignment_per_rel = sg_results[4]
		alignment_accuracies, better_alignments, same_alignments, total_alignments = sg_results[5], sg_results[6], sg_results[7], sg_results[8]
		combo_alignment_acc, combined_better_alignment, combined_same_alignment = sg_results[9], sg_results[10], sg_results[11]

		# Save scores
		metrics.sg_better_score += better_scores
		metrics.sg_total += total_scores

		# Save relation successes
		for rel in successful_relations:
			metrics.sg_evaluation_dict[rel] += 1
		for rel in successful_relations_align:
			metrics.sg_evaluation_dict_align[rel] += 1
		for rel in total_relations:
			metrics.sg_total_dict[rel] += 1

		# Save relations and accuracies (success rates)
		for (rel, acc) in sg_alignment_per_rel:
			metrics.sg_alignment_accuracy[rel].append(acc)

		# Save phrase grounding accuracies
		metrics.default_alignment_accuracies.append(default_alignment_acc)

		for acc in alignment_accuracies:
			metrics.new_alignment_accuracies.append(acc)

		# Better alignment?
		metrics.sg_better_alignment_score += better_alignments
		metrics.sg_same_alignment_score += same_alignments
		metrics.sg_better_alignment_total += total_alignments

		metrics.sg_combined_better_alignment += combined_better_alignment
		metrics.sg_combined_same_alignment += combined_same_alignment
		metrics.sg_combined_total_alignment += 1

		# Get result parts for phrase replacement results
		better_scores, total_scores = phrase_results[0], phrase_results[1]
		successful_relations, total_relations, successful_relations_align = phrase_results[2], phrase_results[3], phrase_results[-1]
		phrase_alignment_per_rel = phrase_results[4]
		alignment_accuracies, better_alignments, same_alignments, total_alignments = phrase_results[5], phrase_results[6], phrase_results[7], phrase_results[8]
		combo_alignment_acc, combined_better_alignment, combined_same_alignment = phrase_results[9], phrase_results[10], phrase_results[11]

		# Save scores
		metrics.phrase_better_score += better_scores
		metrics.phrase_total += total_scores

		# Save relation successes
		for rel in successful_relations:
			metrics.phrase_evaluation_dict[rel] += 1
		for rel in successful_relations_align:
			metrics.phrase_evaluation_dict_align[rel] += 1
		for rel in total_relations:
			metrics.phrase_total_dict[rel] += 1

		# Save relations and accuracies (success rates)
		for (rel, acc) in phrase_alignment_per_rel:
			metrics.phrase_alignment_accuracy[rel].append(acc)

		# Save phrase grounding accuracies
		for acc in alignment_accuracies:
			metrics.new_alignment_accuracies.append(acc)

		# Better alignment?
		metrics.phrase_better_alignment_score += better_alignments
		metrics.phrase_same_alignment_score += same_alignments
		metrics.phrase_better_alignment_total += total_alignments

		metrics.phrase_combined_better_alignment += combined_better_alignment
		metrics.phrase_combined_same_alignment += combined_same_alignment
		metrics.phrase_combined_total_alignment += 1
		
		

def pipeline(file, ONLY_GOOD_RELATIONS, emb_version=False):
	"""Processing pipeline for including alternatives (support concepts)
	for scene graph objects and phrases and evaluating. """

	# Get caption/image IDs
	caption_ID = file.split('_')[-1].strip('.pickle')
	image_ID = file.split('_')[-2]

	# Load entities
	with open('/home/students/suter/ConceptNet/data/entity_IDs/entity_ID_dict_'+image_ID+'_'+caption_ID+'.pickle', 'rb') as handle:
		ent_IDs = pickle.load(handle)

	# Get files to use
	current_files = [f for f in files if f[:-4].split('_')[0] == image_ID
									  and f[:-4].split('_')[1] == caption_ID]

	# Get phrase indices
	phrase_indices = [f[:-4].split('_')[2] for f in current_files]

	# Stop if no phrases
	if not phrase_indices:
		return None

	# Get original performance scores
	sg_object_array, res_matrix, default_adj_matrix, default_score, default_alignment_acc = get_default_scores(files, image_ID, caption_ID, phrase_indices, ent_IDs)

	# Replace with support concepts for scene graph objects, get evaluation
	sg_alt_results = get_sg_alternatives(image_ID, caption_ID, phrase_indices, \
										sg_object_array, res_matrix, default_adj_matrix, \
										default_score, default_alignment_acc, \
										ONLY_GOOD_RELATIONS, ent_IDs, emb_version)

	# Replace with support concepts for phrases, get evaluation
	phrase_alt_results = get_phrase_alternatives(image_ID, caption_ID, phrase_indices, \
												sg_object_array, res_matrix, default_adj_matrix, \
												default_score, default_alignment_acc, \
												ONLY_GOOD_RELATIONS, ent_IDs, emb_version)

	return sg_alt_results, phrase_alt_results, default_alignment_acc


# Start SLURM process
cluster = SLURMCluster(

	queue='main',
	cores=8,
	memory="36000 MB",
	processes = 7,
	job_extra = ['--job-name=alternatives',
				'--mail-type=NONE',
				'-e /home/students/suter/logs_suter/worker_logs/slurm-%j.err',
				'-o /home/students/suter/logs_suter/worker_logs/slurm-%j.out',
				'--time=24:00:00'],

	env_extra = ['PATH=/opt/slurm/bin:$PATH']
)

# Load data

# Which files to use
with open('used_files/used_files_500_incl_alt.txt') as infile:
	files_to_use = infile.read().split('\n')

# Get files in pageranks folder (= ConceptNet process sampels)
files = os.listdir('./.cosim_collection')
pagerank_matrix_files = os.listdir('../ConceptNet/data/pageranks')

# Start metrics
metrics = Metrics()

# Set TAG (which version
TAG = ['_emb_version', ''][0]

# Multi-processing or not
MULTI_PRO = False

# Get client ready
if MULTI_PRO:
	cluster.scale(jobs=8)
	print(cluster.job_script())
	client = Client(cluster)

# Initalize
futures = []
chunk_counter = 0
results =  []

# Processing counters
processed_counter = 0
max_samples = 500


# Iterate over all ready files
for file in pagerank_matrix_files:

	# Get IDs
	parts = file.split('_')
	image_ID = parts[-2].strip()
	caption_ID = parts[-1].strip('.pickle')

	# Only use selected files
	if image_ID not in files_to_use:
		continue

	# Count progress and abort eventually
	processed_counter += 1
	if processed_counter > max_samples:
		break

	# Without multi-processing
	if not MULTI_PRO:

		# Start pipeline with corresponding version
		if TAG == '_emb_version':
			result = pipeline(file, ONLY_GOOD_RELATIONS, emb_version=True)
		else:
			result = pipeline(file, ONLY_GOOD_RELATIONS, emb_version=False)

		results.append(result)

	# With multi-processing 
	else:

		# Start pipeline with corresponding version
		if TAG == '_emb_version':
			future = client.submit(pipeline, file, ONLY_GOOD_RELATIONS, emb_version=True)
		else:
			future = client.submit(pipeline, file, ONLY_GOOD_RELATIONS, emb_version=False)

		# Save results
		futures.append(future)
		
		# Only process chunks of 50 (memory issues)
		if chunk_counter >= 50:
			results = client.gather(futures)
			process_results(results,metrics)
			futures = []
			chunk_counter = 0

# Process final results
if MULTI_PRO:
	results = client.gather(futures)
process_results(results, metrics)

# Initalize
useful_sg_relations = []
useful_phrase_relations = []

sg_relations_acc_dict = {}
phrase_relations_acc_dict = {}

sg_relations_acc_dict_align = {}
phrase_relations_acc_dict_align = {}

# PRINTING RESULTS

print('PHRASE ALTERNATIVES')
for key in metrics.phrase_evaluation_dict.keys():
	acc = metrics.phrase_evaluation_dict[key]/metrics.phrase_total_dict[key]*100
	print('Key:', key)
	print('Success:', metrics.phrase_evaluation_dict[key])
	print('Total:',metrics.phrase_total_dict[key])
	print('Acc:', acc)
	phrase_relations_acc_dict[key] = (acc, metrics.phrase_total_dict[key])
	if acc > 50.0:
		useful_phrase_relations.append(key)

print('PHRASE ALTERNATIVES (align)')
for key in metrics.phrase_evaluation_dict_align.keys():
	acc = metrics.phrase_evaluation_dict_align[key]/metrics.phrase_total_dict[key]*100
	print('Key:', key)
	print('Success:', metrics.phrase_evaluation_dict_align[key])
	print('Total:',metrics.phrase_total_dict[key])
	print('Acc:', acc)
	phrase_relations_acc_dict_align[key] = (acc, metrics.phrase_total_dict[key])


print('\n----------------\n\n')

print('PHRASE ALTERNATIVES ALIGNMENT')
for key in metrics.phrase_alignment_accuracy.keys():
	acc = sum(metrics.phrase_alignment_accuracy[key])/len(metrics.phrase_alignment_accuracy[key])
	print('Key:', key)
	print('Mean Acc:',acc)

print('SG OBJECT ALTERNATIVES')
for key in metrics.sg_evaluation_dict.keys():
	acc = metrics.sg_evaluation_dict[key]/metrics.sg_total_dict[key]*100
	print('Key:', key)
	print('Success:', metrics.sg_evaluation_dict[key])
	print('Total:',metrics.sg_total_dict[key])
	print('Acc:', acc)
	sg_relations_acc_dict[key] = (acc, metrics.sg_total_dict[key])

	if acc > 50.0:
		useful_sg_relations.append(key)

print('SG OBJECT ALTERNATIVES (align)')
for key in metrics.sg_evaluation_dict_align.keys():
	acc = metrics.sg_evaluation_dict_align[key]/metrics.sg_total_dict[key]*100
	print('Key:', key)
	print('Success:', metrics.sg_evaluation_dict_align[key])
	print('Total:',metrics.sg_total_dict[key])
	print('Acc:', acc)
	sg_relations_acc_dict_align[key] = (acc, metrics.sg_total_dict[key])


print('SG ALTERNATIVES ALIGNMENT')
for key in metrics.sg_alignment_accuracy.keys():
	acc = sum(metrics.sg_alignment_accuracy[key])/len(metrics.sg_alignment_accuracy[key])
	print('Key:', key)
	print('Mean Acc:',acc)

print('\n----------------\n\n')

print('TOTAL SG ACCOUNT')
print('Successes:', metrics.sg_better_score)
print('Total:', metrics.sg_total)
print('Acc:', metrics.sg_better_score/metrics.sg_total*100)
print('Better alignment:', metrics.sg_better_alignment_score/metrics.sg_better_alignment_total*100)
print('Same alignment:', metrics.sg_same_alignment_score/metrics.sg_better_alignment_total*100)
print('Combined better aligment:', metrics.sg_combined_better_alignment/metrics.sg_combined_total_alignment*100)
print('Combined same alignment:', metrics.sg_combined_same_alignment/metrics.sg_combined_total_alignment*100)
print('\n----------------\n\n')

print('TOTAL PHRASE ACCOUNT')
print('Successes:', metrics.phrase_better_score)
print('Total:', metrics.phrase_total)
print('Acc:', metrics.phrase_better_score/metrics.phrase_total*100)
print('Better alignment:', metrics.phrase_better_alignment_score/metrics.phrase_better_alignment_total*100)
print('Same alignment', metrics.phrase_same_alignment_score/metrics.phrase_better_alignment_total*100)
print('Combined better aligment:', metrics.phrase_combined_better_alignment/metrics.phrase_combined_total_alignment*100)
print('Combined same alignment:', metrics.phrase_combined_same_alignment/metrics.phrase_combined_total_alignment*100)
print('\n----------------\n\n')

print('DEFAULT ALIGNMENT:')
print(sum(metrics.default_alignment_accuracies)/len(metrics.default_alignment_accuracies))
print('NEW ALIGNMENT:')
print(sum(metrics.new_alignment_accuracies)/len(metrics.new_alignment_accuracies))
print()

print('Processed samples:', processed_counter)


# Save relations and success rates (if new ones are generated)
if not ONLY_GOOD_RELATIONS:

	with open('conceptnet_data'+new_file_tag+'/useful_sg_relations'+TAG+'.pickle', 'wb') as handle:
	   pickle.dump(useful_sg_relations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('conceptnet_data'+new_file_tag+'/useful_phrase_relations'+TAG+'.pickle', 'wb') as handle:
	   pickle.dump(useful_phrase_relations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('conceptnet_data'+new_file_tag+'/sg_relations_acc_dict'+TAG+'.pickle', 'wb') as handle:
	   pickle.dump(sg_relations_acc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('conceptnet_data'+new_file_tag+'/sg_relations_acc_dict_align'+TAG+'.pickle', 'wb') as handle:
	   pickle.dump(sg_relations_acc_dict_align, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('conceptnet_data'+new_file_tag+'/phrase_relations_acc_dict'+TAG+'.pickle', 'wb') as handle:
	   pickle.dump(phrase_relations_acc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('conceptnet_data'+new_file_tag+'/phrase_relations_acc_dict_align'+TAG+'.pickle', 'wb') as handle:
	   pickle.dump(phrase_relations_acc_dict_align, handle, protocol=pickle.HIGHEST_PROTOCOL)
