# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# support_concepts.py
# ----------------
# Support concepts - also called: alternatives - for scene graph objects and phrases;
# integrates support concepts as embeddings into phrase grounding system
# based on pre-computed success rates for relations in ConceptNet.
 
 
import config as cfg


# Initialize
n_alts_phrases = []
n_alts_objects = []
n_all_obj_alternatives = []
n_all_phrase_alternatives = []
alt_phrase_found = []
alt_phrase_total = []
alt_sg_found = []
alt_sg_total = []

n_alt_phrase_found = []
n_alt_phrase_total = []
n_alt_sg_found = []
n_alt_sg_total = []

nn_alt_phrase_found = []
nn_alt_phrase_total = []
nn_alt_sg_found = []
nn_alt_sg_total = []

sg_ratio = []
phrase_ratio  = []


def get_alternative_phrase_embeddings(image_ID, caption_ID, min_acc=cfg.phrase_min_acc, min_count=3, new_objects=True, n_phrases=0):
	"""Get alternative phrases (support concepts) and get new combined word embeddings
	representig old phrase and new phrase(s)."""

	
	# Load alternative phrases
	with open("../ConceptNet/data/alternatives/alternatives_phrase_"+image_ID+"_"+caption_ID+".pickle", "rb") as input_file:
		alternative_phrases = pickle.load(input_file)

	# Load different version
	if cfg.alternative_path == '_new':
		with open("../ConceptNet/data/alternatives_red/alternatives_phrase_"+image_ID+"_"+caption_ID+".pickle", "rb") as input_file:
			alternative_phrases = pickle.load(input_file)

	# Load useful relations
	with open('conceptnet_data'+cfg.alternative_path+'/useful_phrase_relations'+cfg.alt_ver_tag+'.pickle', 'rb') as handle:
	   useful_phrase_relations = pickle.load(handle)
	   
	# Load success rates for relations
	with open('conceptnet_data'+cfg.alternative_path+'/phrase_relations_acc_dict'+cfg.alt_ver_tag+'.pickle', 'rb') as handle:
	   phrase_relations_acc_dict = pickle.load(handle)

	# Load success rates for relations (different version: count accuracy increase as success, not weighted sum)
	with open('conceptnet_data'+cfg.alternative_path+'/phrase_relations_acc_dict_align'+cfg.alt_ver_tag+'.pickle', 'rb') as handle:
	   phrase_relations_acc_dict_align = pickle.load(handle)

	# Select version
	if cfg.acc_over_cosim:
		phrase_relations_acc_dict = phrase_relations_acc_dict_align

	# Get number of alternative phrases
	n_alternative_phrases_org =  len(set([tuple(relation) for (phrase_ID, old_label, new_label, relation, _, path_length) in alternative_phrases]))
	n_pres_alternative_phrases = len(set([tuple(relation) for (phrase_ID, old_label, new_label, relation, _, path_length) in alternative_phrases if tuple(relation) in phrase_relations_acc_dict.keys()]))

	# Filter relations and alternative phrases based on success rates and occurrences
	useful_phrase_relations = [rel for rel in phrase_relations_acc_dict.keys() if phrase_relations_acc_dict[rel][0] >= min_acc and phrase_relations_acc_dict[rel][1] > min_count]
	alternative_phrases = [(phrase_ID, old_label, new_label, relation, _, path_length) for (phrase_ID, old_label, new_label, relation, _, path_length) in alternative_phrases if tuple(relation) in useful_phrase_relations]

	# Count ratio of selected alternatives
	if n_alternative_phrases_org > 0:
		phrase_ratio.append(n_pres_alternative_phrases/n_alternative_phrases_org*100)

	# Counting
	if len(alternative_phrases)>0:
		alt_phrase_found.append(1)
	alt_phrase_total.append(1)

	n_alt_phrase_found.append(len(alternative_phrases))
	n_alt_phrase_total.append(n_phrases)
	nn_alt_phrase_total.append(n_phrases)

	# Initialize
	new_phrase_dict = {}
	new_combo_phrase_emb = {}
	old_phrase_index = {}
	phrase_idx_dict = {}
	new_label_by_idx = {}
	filtered_alternatives = {}
	
	all_relevant_alternatives = []
	n_alts_per_phrase = []
	col = []

	# Generate dict for support phrases
	for phrase_ID, old_label, new_label, relation, _, _ in alternative_phrases:
		filtered_alternatives[old_label] = []

	# For each alternative check whether relation has high enough success rate
	for phrase_ID, old_label, new_label, relation, _, _ in alternative_phrases:
		relation = tuple(relation)
		if relation in phrase_relations_acc_dict.keys():
			if [phrase_ID, old_label, new_label, relation, phrase_relations_acc_dict[relation][0]] not in filtered_alternatives[old_label]:
				filtered_alternatives[old_label].append([phrase_ID, old_label, new_label, relation, phrase_relations_acc_dict[relation][0]])

	# For each old label, get suitable new labels (with high sucess rate)
	for old_label in filtered_alternatives.keys():
		
		# Get all new labels
		alternatives_per_label = filtered_alternatives[old_label]
		# Sort by success rate
		sorted_alternatives = sorted(alternatives_per_label, key=lambda x:x[4], reverse=True)[:cfg.number_of_alts_phrase]
		
		# Save all relevant altenratives
		all_relevant_alternatives += sorted_alternatives
		
		# Counters
		n_alts_per_phrase.append(len(alternatives_per_label))
		n_all_phrase_alternatives.append(len(alternatives_per_label))
		col.append(len(alternatives_per_label))

	# For each phrase, get generated new dict for alternative phrases
	for phrase_ID, old_label, new_label, relation, acc in all_relevant_alternatives:
		new_phrase_dict[old_label] = []
		new_label_by_idx[phrase_ID] = []

	# For each old phrase, save new phrases and embeddings by old label and index
	for phrase_ID, old_label, new_label, _, _ in all_relevant_alternatives:
		new_phrase_dict[old_label].append(new_label)
		phrase_idx_dict[old_label] = phrase_ID
		new_label_by_idx[phrase_ID].append(emb.embed(new_label))

	# For each old phrase, generate the combo embedding with old and new phrases
	for ol_idx,old_label in enumerate(new_phrase_dict.keys()):
		
		# Get new labels
		new_labels = new_phrase_dict[old_label]
		old_phrase_index[old_label] = ol_idx
		
		# Get shape for combo array
		new_shape = [len(new_labels)+1]+list(emb.embed(new_labels[0]).shape)

		# Generate combo array and add old phrase embedding
		combo_emb_array = np.zeros(tuple(new_shape))
		combo_emb_array[0,:] = emb.embed(old_label)
		
		# Add new phrases (support concepts)
		for label_idx, new_label in enumerate(new_labels):
			combo_emb_array[label_idx+1,:] = emb.embed(new_label)

		# Get mask for phrases with no embedding
		zero_mask = np.isclose(combo_emb_array, 0.0)
		sample_mask = np.all(zero_mask, axis=1)
		
		# Only keep those with found embedding
		sample_mask = ~sample_mask		
		combo_emb_array = combo_emb_array[sample_mask]

		# Get mean of embeddings
		combo_emb_array = combo_emb_array.mean(axis=0)

		# Get double embedding array with only old phrase and 1 new phrase
		double_emb_array = np.zeros((2, emb.embed(new_labels[0]).shape[0]))
		double_emb_array[0,:] = emb.embed(old_label)
		double_emb_array[1,:] = emb.embed(new_label)
		
		# Get mean for double embedding array
		double_emb_array = double_emb_array.mean(axis=0)

		# Save new embeddings
		phrase_idx = phrase_idx_dict[old_label]
		new_combo_phrase_emb[phrase_idx] = combo_emb_array
		new_combo_phrase_emb[old_label] = double_emb_array

	# Counters
	n_alts_phrases.append(len(all_relevant_alternatives))
	nn_alt_phrase_found.append(sum(col))

	return new_combo_phrase_emb, new_label_by_idx


def get_alternative_sg_embeddings(original_sg_objects, image_ID, caption_ID, rgb_dict, min_acc=cfg.object_min_acc, min_count=3, new_objects=False):
	"""Include alternative scene graph objects (support concepts) and return updated objects with new embeddings and labels."""
 
	# Load alternative scene graph objects
	with open("../ConceptNet/data/alternatives/alternatives_sg_"+image_ID+"_"+caption_ID+".pickle", "rb") as input_file:
		alternative_sg_objs = pickle.load(input_file)

	# Load different version
	if cfg.alternative_path == '_new':
		with open("../ConceptNet/data/alternatives_red/alternatives_sg_"+image_ID+"_"+caption_ID+".pickle", "rb") as input_file:
			alternative_sg_objs = pickle.load(input_file)

	# Load useful relations
	with open('conceptnet_data'+cfg.alternative_path+'/useful_sg_relations'+cfg.alt_ver_tag+'.pickle', 'rb') as handle:
	   useful_sg_relations = pickle.load(handle)

	# Load success rates for relations
	with open('conceptnet_data'+cfg.alternative_path+'/sg_relations_acc_dict'+cfg.alt_ver_tag+'.pickle', 'rb') as handle:
	   sg_relations_acc_dict = pickle.load(handle)

	# Load success rates for relations (different version: count accuracy increase as success, not weighted sum)
	with open('conceptnet_data'+cfg.alternative_path+'/sg_relations_acc_dict_align'+cfg.alt_ver_tag+'.pickle', 'rb') as handle:
	   sg_relations_acc_dict_align = pickle.load(handle)

	# Select version
	if cfg.acc_over_cosim:
		sg_relations_acc_dict = sg_relations_acc_dict_align

	# Get number of alternative objects
	n_alternative_sg_org =  len(set([tuple(relation) for (phrase_ID, old_label, new_label, relation, _, path_length) in alternative_sg_objs]))
	n_pres_alternative_sg = len(set([tuple(relation) for (phrase_ID, old_label, new_label, relation, _, path_length) in alternative_sg_objs if tuple(relation) in sg_relations_acc_dict.keys()]))

	# Filter relations and alternative objects based on success rates and occurrences
	useful_sg_relations = [rel for rel in sg_relations_acc_dict.keys() if sg_relations_acc_dict[rel][0] >= min_acc and sg_relations_acc_dict[rel][1] >= min_count]
	alternative_sg_objs = [(phrase_ID, old_label, new_label, relation, _, path_length) for (phrase_ID, old_label, new_label, relation, _, path_length) in alternative_sg_objs if tuple(relation) in useful_sg_relations]

	# Count ratio of selected alternatives
	if n_alternative_sg_org > 0:
		sg_ratio.append(n_pres_alternative_sg/n_alternative_sg_org*100)

	# Counting
	if len(alternative_sg_objs)>0:
		alt_sg_found.append(1)
	alt_sg_total.append(1)
	
	n_alt_sg_found.append(len(alternative_sg_objs))
	n_alt_sg_total.append(len(original_sg_objects[0]))
	nn_alt_sg_total.append(len(original_sg_objects[0]))

	# Initialize
	new_label_dict = {}
	new_combo_label_emb = {}
	old_label_index = {}

	filtered_alternatives = {}
	all_relevant_alternatives = []
	n_alts_per_object = []
	col = []

	# Generate dict for support phrases
	for phrase_ID, old_label, new_label, relation, _, _ in alternative_sg_objs:
		filtered_alternatives[old_label] = []

	# For each alternative check whether relation has high enough success rate
	for phrase_ID, old_label, new_label, relation, _, _ in alternative_sg_objs:
		relation = tuple(relation)
		if relation in sg_relations_acc_dict.keys():
			if [phrase_ID, old_label, new_label, relation, sg_relations_acc_dict[relation][0]] not in filtered_alternatives[old_label]:
				filtered_alternatives[old_label].append([phrase_ID, old_label, new_label, relation, sg_relations_acc_dict[relation][0]])

	# For each old label, get suitable new labels (with high sucess rate)
	for old_label in filtered_alternatives.keys():
		
		# Get alternative labels
		alternatives_per_label = filtered_alternatives[old_label]

		# Sort labels
		sorted_alternatives = sorted(alternatives_per_label, key=lambda x:x[4], reverse=True)[:cfg.number_of_alts]

		# Save all relevant alternative labels
		all_relevant_alternatives += sorted_alternatives
		
		# Counters
		n_alts_per_object.append(len(sorted_alternatives))
		n_all_obj_alternatives.append(len(alternatives_per_label))
		col.append(len(alternatives_per_label))

	# For each label, get generated new dict for alternative labels
	for phrase_ID, old_label, _, _, _ in all_relevant_alternatives:
		new_label_dict[old_label] = []

	# For each old label, save new labels and embeddings by old label and index
	for phrase_ID, old_label, new_label, _, _  in all_relevant_alternatives:
		new_label_dict[old_label].append(new_label)

	# For each old scene graph label, generate the combo embedding with old and new labels
	for ol_idx, old_label in enumerate(new_label_dict.keys()):
		
		# Get new labels and index
		old_label_index[old_label] = ol_idx
		new_labels = new_label_dict[old_label]
		
		# Generate new combo array
		new_shape = [len(new_labels)+1]+list(emb.embed(new_labels[0]).shape)
		combo_emb_array = np.zeros(tuple(new_shape))

		# Add old label to combo array
		combo_emb_array[0,:] = emb.embed(old_label)
		
		# Add new labels (support concepts)
		for label_idx, new_label in enumerate(new_labels):
			combo_emb_array[label_idx+1,:] = emb.embed(new_label)
		
		# Get double embedding array with only old label and 1 new label
		double_emb_array = np.zeros((2, emb.embed(new_labels[0]).shape[0]))
		double_emb_array[0,:] = emb.embed(old_label)
		double_emb_array[1,:] = emb.embed(new_label)
		
		# Get mean for double embedding array
		double_emb_array = double_emb_array.mean(axis=0)

		# Get mask for phrases with no embedding
		zero_mask = np.isclose(combo_emb_array, 0.0)
		sample_mask = np.all(zero_mask, axis=1)
		
		# Only keep those with found embedding		
		sample_mask = ~sample_mask
		combo_emb_array = combo_emb_array[sample_mask]

		# Get mean of embeddings
		combo_emb_array = combo_emb_array.mean(axis=0)
		
		# Save new combo embeddings
		new_combo_label_emb[old_label] = combo_emb_array
		new_combo_label_emb[old_label] = double_emb_array

	# Counters
	n_alts_objects.append(sum(n_alts_per_object))
	nn_alt_sg_found.append(sum(col))

	# Initalize
	new_embeddings = []
	sg_objects = [[],[],[]]
	extended_sg_objects = [[],[],[]]

	# For each object save new embeddings in scene graph objects
	for i, label in enumerate(original_sg_objects[0]):
		
		# Get label and ID
		cat, ID = label.split('-')

		# Append label, coordinates and embeddings
		extended_sg_objects[0].append(original_sg_objects[0][i])
		extended_sg_objects[1].append(original_sg_objects[1][i])
		extended_sg_objects[2].append(original_sg_objects[2][i])

		# For each category, save new objects
		if cat in new_label_dict.keys():
		
			# Get old label and embedding
			old_label = cat
			obj_emb = new_combo_label_emb[old_label]
			
			# Save combo embedding and labels
			new_embeddings.append(obj_emb)
			new_labels = new_label_dict[old_label]
			
			# For each new label
			for new_label in new_labels:
				
				# Get new embedding and save as scene graph object
				new_emb = emb.embed(new_label)
				extended_sg_objects[0].append(new_label+'-'+ID) # label
				extended_sg_objects[1].append(new_emb)	# embedding
				extended_sg_objects[2].append(original_sg_objects[2][i]) # coordindates
				
				# Make new entry in color dict
				rgb_dict[new_label+'-'+ID] = rgb_dict[cat+'-'+ID]

		else:
			# Take original embedding if there is no new label
			new_embeddings.append(original_sg_objects[1][i])


	# Overwrite scene graph objects with new embeddings
	sg_objects[0] = original_sg_objects[0]
	sg_objects[1] = new_embeddings
	sg_objects[2] = original_sg_objects[2]

	# If new objects are generated rather than updated embeddings, keep original objects
	if new_objects:
		sg_objects = extended_sg_objects

	
	return sg_objects, rgb_dict


def print_statistics():
	"""Print saved statistics for support concepts."""

	print('Alternatives phrases', sum(n_alts_phrases)/len(n_alts_phrases))
	print('Alternatives objcts', sum(n_alts_objects)/len(n_alts_objects))
	print('alternative objects per label', sum(n_all_obj_alternatives)/len(n_all_obj_alternatives))
	print('Alternative phrases per label', sum(n_all_phrase_alternatives)/len(n_all_phrase_alternatives))
	print()
	print('Alt phrase found', sum(alt_phrase_found))
	print('Alt phrase total', sum(alt_phrase_total))
	print('Alt sg found', sum(alt_sg_found))
	print('Alt sg total', sum(alt_sg_total))
	print()
	print('N Alt phrase found', sum(n_alt_phrase_found))
	print('N Alt phrase total', sum(n_alt_phrase_total))
	print('N Alt sg found', sum(n_alt_sg_found))
	print('N Alt sg total', sum(n_alt_sg_total))
	print()
	print('NN Alt phrase found', sum(nn_alt_phrase_found))
	print('NN Alt phrase total', sum(nn_alt_phrase_total))
	print('NN Alt sg found', sum(nn_alt_sg_found))
	print('NN Alt sg total', sum(nn_alt_sg_total))
	print()
	print('SG ratio', sum(sg_ratio)/len(sg_ratio))
	print('SG phrase', sum(phrase_ratio)/len(phrase_ratio))
