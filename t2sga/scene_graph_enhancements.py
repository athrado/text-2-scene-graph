# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# scene_graph_enhancements.py
# ----------------
# Various enhancements of the scene graph represention:
# - remove duplicate nodes
# - clean labels
# - remove multiple representations (including parameter screening)
# - add plural objects
# - add background attribute
# - add color attribute


# Imports
import shutil
import os
import re
import timeit
import collections
import itertools

from PIL import Image
import numpy as np
import scipy
import pattern.en as pat
import webcolors

# Import own modules
import config as cfg
import evaluate
import util.graph_functions as graph_functions
import util.get_best_label as labelgetter
import comparison_with_gold as gold_comp
import Scene_Graph as sg
import util.colors as colors


# Reduction tracking
overall_reduction_rate = []
overall_people_reduction_rate = []
overall_object_reduction_rate = []

unknown_label_counter = 0
all_label_counter = 0

# Combination dicts (exemplary)
combi_dict = {('boy','girl'):'child-pl',
			  ('man','woman'):'couple-pl',
			  ('cat','dog'):'animal-pl' }


def clean_labels(file_ID, target_dir):
	"""Remove plural objects (people, men) and fix singular labels."""

	# Get the Scene Graph for image
	scene_graph = sg.SceneGraph(file_ID, sg_data_dir=target_dir)

	# Read the bounding boxes file
	with open(scene_graph.bboxes_file) as infile:
		data = infile.read()

	# Get lines (without empty lines)
	lines = data.split('\n')
	lines = [line for line in lines if line.strip() != '']

	data = []

	# Fix singular labels and remove plural labels
	for line in lines:

		# Get label and ID
		full_label, coords = line.split(';')
		full_label = full_label.strip()
		label, ID = full_label.split('-')

		# Remove plural objects
		if label in ['people','men']:
			continue

		# Add "s" to certain singular labels
		if label in ['jean','short','trunk','pant','trouser']:
			label += 's'

		# Save new data
		new_line = label+'-'+ID+' ; '+coords
		data.append(new_line)

	data = '\n'.join(data)

	# Overwrite current bboxes file with new one
	with open(target_dir+file_ID+'_bboxes.txt','w') as outfile:
		outfile.write(data)
		outfile.write('\n')

	# Read the triple (relations) file
	with open(scene_graph.res_file) as infile:
		data = infile.read()

	# Get lines (without empty lines)
	lines = data.split('\n')
	lines = [line for line in lines if line.strip() != '']

	data = []

	# Fix singular labels and remove plural labels
	for line in lines:

		# Get nodes
		node1, rel, node2 = line.split(' - ')
		node1 = node1.strip().split('-')[0]
		node2 = node2.strip().split('-')[0]

		# Remove plural objects
		if node1 in ['people','men'] or node2 in ['people','men']:
			continue

		# Fix some singular labels
		line = re.sub('jean-','jeans-',line)
		line = re.sub('short-','shorts-',line)
		line = re.sub('pant-','pants-',line)
		line = re.sub('trunk-','trunks-',line)
		line = re.sub('trouser-','trousers-',line)
		data.append(line)

	data = '\n'.join(data)

	# Overwrite current triple (relations) file with new one
	with open(target_dir+file_ID+'_res.txt','w') as outfile:
		outfile.write(data)
		outfile.write('\n')


def remove_duplicate_nodes(file_ID, target_dir):
	"""Remove duplicate nodes as sometimes found in flickr_all."""

	# Get the Scene Graph for image
	scene_graph = sg.SceneGraph(file_ID, sg_data_dir=target_dir)

	# Read the bounding boxes file
	with open(scene_graph.bboxes_file) as infile:
		data = infile.read()

	# Get lines (without empty lines)
	lines = data.split('\n')
	lines = [line for line in lines if line.strip() != '']

	# Is there duplicate section?
	is_double = False
	if len(lines)%2 == 0:
		# Get middle middle_index
		middle_index = int(len(lines)/2)
		# First and middle index the same?
		if lines[0] == lines[middle_index]:
			is_double = True

	# Get only first half if duplicate
	if is_double:
		first_half = lines[:middle_index]
		data = "\n".join(first_half)

	# Overwrite current bboxes file with new one
	with open(target_dir+file_ID+'_bboxes.txt','w') as outfile:
		outfile.write(data)
		outfile.write('\n')


def add_background_attribute(file_ID, target_dir, tag=''):
	"""Add background attribute."""

	# Get the Scene Graph for image
	scene_graph = sg.SceneGraph(file_ID, sg_data_dir=target_dir)

	# Compute ratio of original image to downsampled image (1024, 768, 3)
	original_image = scene_graph.original_image

	# Get scene graph nodes
	scene_graph.get_nodes()

	person_nodes = ['woman','man','girl','boy','person','child','kid','lady','guy']
	background_relations = []

	# Get singular people nodes
	singular_nodes = [node for node in scene_graph.nodes if not node.label.endswith('pl')]
	singular_person_nodes = [node for node in singular_nodes if node.category in person_nodes]

	if singular_person_nodes:

		# Get biggest node label, biggest node and its size
		biggest_node_label = graph_functions.get_biggest_node_label(singular_person_nodes)
		biggest_node = [node for node in scene_graph.nodes if node.label == biggest_node_label][0]
		biggest_node_size = graph_functions.get_box_size(biggest_node.coordinates)

		# For each people node, check whether it is in the background
		for node in scene_graph.nodes:
			if node.category in person_nodes:

				# Get bounding box size
				size = graph_functions.get_box_size(node.coordinates)

				# Tag as background if 5 times smaller than biggest person node
				if size < biggest_node_size/5:
					background_relations.append(node.label+' - is_background - True')


	# Save background edges in outfile
	with open(target_dir+file_ID+'_res'+tag+'.txt','a') as outfile:

		out_data = "\n".join(background_relations)
		outfile.write(out_data)
		outfile.write('\n')


def add_color_attribute(file_ID, target_dir, tag=''):
	"""Add color attribute to all nodes."""

	# Get Scene Graph and nodes
	scene_graph = sg.SceneGraph(file_ID, sg_data_dir=target_dir)
	bboxes_file = scene_graph.bboxes_file
	scene_graph.get_nodes()

	# Get original image and ratio
	original_image = scene_graph.original_image
	ratio, _ = scene_graph.get_image_ratio()

	color_relations = []

	# For each node, add color attribute
	for node in scene_graph.nodes:

		# Get node area in original image
		area = node.coordinates
		area = [a*ratio for a in area]

		# Crop and reshape as array
		img = Image.open(original_image)
		cropped_img = img.crop(area)
		ar = np.asarray(cropped_img)
		shape = ar.shape
		ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

		# Get rgb triple by kmeans or median computation
		kmeans_rgb = colors.get_kmeans_rgb(ar)

		# Make it "brighter"
		brighter_computed_rgb = tuple([c+20 for c in kmeans_rgb])

		# Get color name
		color_name = colors.get_colour_name(brighter_computed_rgb)

		# Save colors as edges
		color_relations.append(node.label+' - has_color - '+ color_name)
		color_relations.append(node.label+' - has_rgb - ('+",".join([str(c) for c in brighter_computed_rgb])+')')

	# Save color edges in outfile
	with open(target_dir+file_ID+'_res'+tag+'.txt','a') as outfile:
		out_data = "\n".join(color_relations)
		outfile.write(out_data)
		outfile.write('\n')


def remove_multiple_representations(file_ID, target_dir, iou_threshold_p=0.8, iou_threshold_o=0.4, \
neighbor_threshold_p=0.3, neighbor_threshold_o=0.3, smallerBox_theshold_p=1.1,\
smallerBox_theshold_o=1.1, merge_bboxes=True, merge_labels=True, tag=''):

	# Get Scen Graph and nodes
	scene_graph = sg.SceneGraph(file_ID, sg_data_dir=target_dir)
	scene_graph.get_nodes()
	scene_graph.create_network()

	# Option for when not merging labels (only keep one per label)
	reduce_duplicates = False

	# Get category dict (list of nodes per category)
	label_dict = scene_graph.get_label_dict()

	# Set person nodes
	person_nodes = ['woman','man','girl','boy','person','child','kid','lady','guy']
	similar_object_nodes = ['truck','car','motorcycle','bike','book','paper','glass',
							'cup','laptop','phone','table','desk','stand']

	# Initialize
	merge_edges = []
	merge_dict = {}
	cat_counter_dict = {}

	# For each label, get maximum ID (equals number of nodes per category)
	for cat in graph_functions.object_cats:
		if cat in label_dict.keys():
			cat_counter_dict[cat] = len(label_dict[cat])
		else:
			cat_counter_dict[cat] = 0

	# Get number of original nodes
	n_original_nodes = len(scene_graph.nodes)
	n_original_people_nodes = len([node for node in scene_graph.nodes if node.category in person_nodes])
	n_original_object_nodes = len([node for node in scene_graph.nodes if node.category not in person_nodes])

	# For each label category, find overlapping boxes
	for cat in label_dict.keys():

		# Skip person nodes (as they are treated under with the person_ label)
		if cat in person_nodes+similar_object_nodes:
			continue

		# If label is person_, collect all people nodes
		if cat == 'person_':

				 # Get nodes of person category and sort
				 cat_nodes = [node for node in scene_graph.nodes if node.category in person_nodes and not node.label.endswith('pl')]
				 cat_nodes.sort(key=lambda x: x.rank, reverse=False)

		else:

				# Get nodes of given label set (e.g. truck/car; phone/computer)
				if cat.endswith('_'):

					# Get all nodes with this label/cat
					all_cats = list(set([node.category for node in label_dict[cat]]))
					cat_nodes = [node for node in scene_graph.nodes if node.category in all_cats]

				# Get nodes of given label
				else:
					cat_nodes = [node for node in scene_graph.nodes if node.category == cat]

				# Sort by ID
				cat_nodes.sort(key=lambda x: x.rank, reverse=False)

		# For each node pair, predict whether they represent same person/object
		for node1, node2 in itertools.combinations(cat_nodes,2):

				# Skip if same node
				if node1.label == node2.label:
					continue

				# Compute intersection-over-union scores
				iou_score = graph_functions.iou(node1.coordinates, node2.coordinates)
				ioSmallerBox = graph_functions.intersection_over_smallerBox(node1.coordinates, node2.coordinates)

				# Compute neighbor overlap score
				if cat == 'person_':
					neighbor_overlap_score = graph_functions.neighbor_overlap(node1, node2, bodyparts=True)
				else:
					neighbor_overlap_score = graph_functions.neighbor_overlap(node1, node2, bodyparts=False)

				# ioSmallerBox: is not used here

				# Check whether to merge edges (people nodes)
				if cat == 'person_':
					if (neighbor_overlap_score >= neighbor_threshold_p and iou_score >= iou_threshold_p) or ioSmallerBox >= smallerBox_theshold_p:
						merge_edges.append((node1.label, node2.label))

				# Check whether to merge eges (object nodes)
				if cat != 'person_':
					if (neighbor_overlap_score >= neighbor_threshold_o and iou_score >= iou_threshold_o) or ioSmallerBox >= smallerBox_theshold_o:
						merge_edges.append((node1.label, node2.label))

	# Get connected components for the SG based on merge edges
	connected_components = graph_functions.get_connected_components(merge_edges)

	del_nodes = []

	# If bboxes representing same node are to be merged
	if merge_bboxes:

		# For each connected components group
		for group in connected_components.keys():

			# Get nodes
			group_nodes = [node for node in scene_graph.nodes if node.label in connected_components[group]]

			# Set base coordinates (first node)
			merged_bboxes = group_nodes[0].coordinates

			# Get all coordinates for this group and merge
			merge_coordinates = [node.coordinates for node in group_nodes]
			merged_bboxes = graph_functions.multiple_box_union(merge_coordinates)

			# Assign merged coordinates to all nodes in this group
			for node in group_nodes:
				node.coordinates = merged_bboxes

			# Option: if the labels are not to be altered, only keep one node each
			if reduce_duplicates and not merge_labels:

				# Sort by ID/rank
				used_cats = []
				group_nodes.sort(key=lambda x: x.rank, reverse=False)

				# For each group node, only save one each (add others to delete list)
				for node in group_nodes:
					if node.category in used_cats:
						del_nodes.append(node.label)
					else:
						used_cats.append(node.category)


	# If selecting best label for group of nodes representing same person/object
	if merge_labels:

		# For each connected components group
		for group in connected_components.keys():

			# Get nodes
			group_nodes = [node for node in scene_graph.nodes if node.label in connected_components[group]]
			group_labels = [node.category for node in group_nodes]
			group_nodes.sort(key=lambda x: x.rank, reverse=False)

			# Get best label for non-people objects
			if group_labels[0] not in person_nodes:

				# Compute most common label (not used)
				label_Counter = collections.Counter(group_labels)
				most_common_label = label_Counter.most_common(1)
				most_common_label = most_common_label[0][0]

				# Get biggest node label
				first_label = graph_functions.get_biggest_node_label(group_nodes)

			# Get best label for people objects
			else:

				# Sort by ID
				group_nodes.sort(key=lambda x: x.rank, reverse=False)

				# Various ways of selecting label: lowest iD, frequency, most specific, general (not used)
				first_label = group_nodes[0].label
				first_label_cat = labelgetter.get_most_suitable_label_by_freq(group_labels)
				first_label_cat = labelgetter.get_most_specific_general_label(group_labels)[0]

				# Get biggest node label
				first_label = graph_functions.get_biggest_node_label(group_nodes)

				# Counters
				global all_label_counter
				all_label_counter += 1

				# Sort by ID
				used_cats = []
				group_nodes.sort(key=lambda x: x.rank, reverse=False)

				# For each group node, only keep one
				for node in group_nodes:
					if node.category in used_cats:
						del_nodes.append(node.label)
					else:
						used_cats.append(node.category)

			# If bounding boxes are to be merged
			if merge_bboxes:

				# Remove all nodes that are not first label
				for node in group_nodes:
					if node.label != first_label:
						del_nodes.append(node.label)

				# Save best label for each node label (for renaming label later)
				for node in group_nodes:
					merge_dict[node.label] = first_label

			# If bounding boxes are not to be merged (not used)
			else:

				# Get label category
				label_cat = first_label.split('-')[0]

				# For each node in the group (except best label),
				# assign a new (lowest available) ID
				for node_i, node in enumerate(group_nodes):
					if node.label != first_label:
						cat_counter_dict[label_cat] += 1
						label_ID = cat_counter_dict[label_cat]
						node.label = label_cat+'-'+str(label_ID)

	# Get final nodes (that are not to be deleted)
	final_nodes = [node for node in scene_graph.nodes if node.label not in del_nodes]

	# Get number of reduced nodes
	n_reduced_people_nodes = len([node for node in final_nodes if node.category in person_nodes])
	n_reduced_object_nodes = len([node for node in final_nodes if node.category not in person_nodes])

	# Get reduction rates
	n_reduced_nodes = len(final_nodes)
	n_reduction_rate = 1-(n_reduced_nodes/n_original_nodes)

	# Edge cases reduction rate (people)
	if n_original_people_nodes>0:
		n_people_reduction_rate = 1-(n_reduced_people_nodes/n_original_people_nodes)
	else:
		n_people_reduction_rate = 0

	# Edge cases reduction rate (objects)
	if n_original_object_nodes>0:
		n_object_reduction_rate = 1-(n_reduced_object_nodes/n_original_object_nodes)
	else:
		n_object_reduction_rate = 0

	# Get lines to write to outfile
	write_out_lines = [node.label+ ' ; ('+", ".join([str(c) for c in node.coordinates])+')' for node in final_nodes]

	# Write to outfile
	with open(target_dir+file_ID+'_bboxes'+tag+'.txt','w') as outfile:
		out_data = "\n".join(write_out_lines)
		outfile.write(out_data)
		outfile.write('\n')

	# Which nodes were replaced/altered
	replaced_nodes = merge_dict.keys()

	triples_updated = []

	# For each triple, replace all node labels with selected label for group
	for triple in scene_graph.all_triples:
		n1, rel, n2 = triple
		if n1 in replaced_nodes:
			n1 = merge_dict[n1]
		if n2 in replaced_nodes:
			n2 = merge_dict[n2]

		# Updated triple
		triples_updated.append(n1 + ' - '+ rel + ' - ' + n2)

	# Save relation triples
	with open(target_dir+file_ID+'_res'+tag+'.txt','w') as outfile:

		out_data = "\n".join(triples_updated)
		outfile.write(out_data)
		outfile.write('\n')

	# Save reduction rates
	overall_reduction_rate.append(n_reduction_rate)
	overall_people_reduction_rate.append(n_people_reduction_rate)
	overall_object_reduction_rate.append(n_object_reduction_rate)


def add_plural_objects(file_ID, target_dir):
	"""Add plural object nodes to the scene graph representation."""

	clean_labels(file_ID, target_dir)

	# Get Scene Graphs
	scene_graph = sg.SceneGraph(file_ID, sg_data_dir=target_dir)
	scene_graph.get_nodes()
	additional_boxes = []

	# Get category dict
	label_dict = scene_graph.get_label_dict()

	# Get people nodes
	people_nodes = [node for node in scene_graph.nodes if node.category \
					in ['person','man','woman','girl','boy','child','kid','lady','guy']]

	# Get male, female and children nodes
	male_nodes = [node for node in scene_graph.nodes if node.category in ['man','guy','boy']]
	female_nodes = [node for node in scene_graph.nodes if node.category in ['woman','girl','lady']]
	child_nodes = [node for node in scene_graph.nodes if node.category in ['child','kid']]

	# Get node categories that have multiples
	multiples = [cat for cat in label_dict.keys() if len(label_dict[cat]) > 1]

	# For each category for which exist multiple nodes
	for cat in multiples:
		# Skip person_ category (used elsewhere)
		if cat.endswith('person_') or cat.endswith('_'):
			continue
		# Get plural box for this category
		bbox = get_plural_box(cat, label_dict[cat])
		additional_boxes.append(bbox)

	# Get male plural box
	if male_nodes and len([node.category for node in male_nodes])>1:
		male_plural = get_plural_box('man', male_nodes, new_label='male')
		additional_boxes.append(male_plural)

	# Get female plural box
	if female_nodes and len([node.category for node in female_nodes])>1:
		female_plural = get_plural_box('woman', female_nodes, new_label='female')
		additional_boxes.append(female_plural)

	# Get children plural box
	if child_nodes and len([node.category for node in child_nodes])>1:
		child_plural = get_plural_box('child', child_nodes, new_label='children')
		additional_boxes.append(child_plural)

	# Get people plural box
	if people_nodes and len([node.category for node in people_nodes])>1:
		all_plurals = get_plural_box('person', people_nodes, new_label='people')
		additional_boxes.append(all_plurals)

	# Get combinational plural boxes (e.g. couple for man+woman; experimental)
	additional_boxes += get_node_combinations(scene_graph.nodes, label_dict)

	# Add new plural objects to representation
	if additional_boxes:

		# Add plural boxes
		with open(target_dir+file_ID+'_bboxes.txt','a') as outfile:
			outfile.write('\n')
			out_data = "\n".join(additional_boxes)
			outfile.write(out_data)
			outfile.write('\n')


def get_plural_box(cat, nodes, max_boxes=False, new_label=None):
	"""Get plural box for given category/label.
	max_boxes: used for limiting number of nodes used per category."""

	# Get min and max values for nodes

	# If max_boxes is set, only take limited number of boxes for merging
	if max_boxes:
		ymax = max([node.ymax for node in nodes if int(node.ID) <= max_boxes])
		ymin = min([node.ymin for node in nodes if int(node.ID) <= max_boxes])
		xmax = max([node.xmax for node in nodes if int(node.ID) <= max_boxes])
		xmin = min([node.xmin for node in nodes if int(node.ID) <= max_boxes])
	else:
		ymax = max([node.ymax for node in nodes])
		ymin = min([node.ymin for node in nodes])
		xmax = max([node.xmax for node in nodes])
		xmin = min([node.xmin for node in nodes])

	# Get new (merged) coordinates
	coodindates = [xmin, ymin, xmax, ymax]

	# Get plural label (pluralize -- not used here!)
	plural_label = pat.pluralize(cat)

	# Define the bounding box entry with new label
	if new_label:
		bbox_line = new_label+'-pl ; ('+", ".join([str(c) for c in coodindates])+')'
	else:
		# Create bounding box line
		bbox_line = cat+'-pl ; ('+", ".join([str(c) for c in coodindates])+')'

	return bbox_line


def get_node_combinations(nodes, label_dict):
	"""Get new boxes for node combinations (experimental).
	e.g. woman+man = couple"""

	new_boxes = []

	# For required categories in combination dict
	for required_cats in combi_dict.keys():

		combi_found = True

		# Check whether required categories are present
		for cat in required_cats:
			# If one cat not present in category dict, discard
			if cat not in label_dict.keys():
				combi_found = False

		# If combination found
		if combi_found:

			# Get nodes required for this node
			combi_nodes = [node for node in nodes if node.category in required_cats]

			# Get min and max coordinates
			xmin = min([node.xmin for node in combi_nodes])
			ymin = min([node.ymin for node in combi_nodes])
			xmax = max([node.xmax for node in combi_nodes])
			ymax = max([node.ymax for node in combi_nodes])

			# Set new merged coordinates
			coodindates = [xmin, ymin, xmax, ymax]

			# Get box line
			bbox_line = combi_dict[required_cats]+' ; ('+", ".join([str(c) for c in coodindates])+')'

			# Append to new boxes
			new_boxes.append(bbox_line)

	return new_boxes


def copy_files(source_dir, target_dir, n=None, files_to_use=None, copy_res_files=False):
	"""Copy bbox and res files (in order to create new representation)."""

	# Get all bounding box files
	bboxes_files = [file for file in os.listdir(source_dir) if file.endswith('bboxes.txt')]

	# Only get n files
	if n:
		bboxes_files = bboxes_files[:n]

	# If files_to_use, only use specific files
	for i, file in enumerate(bboxes_files):

		# Discard if ID not in files_to_use
		if files_to_use is not None and file[:-11] not in files_to_use:
			continue

		# Copy from source to target dir
		shutil.copyfile(source_dir+file, target_dir+file)

	# Copy also res files
	if copy_res_files:

		# Get res files
		res_files = [file for file in os.listdir(source_dir) if file.endswith('res.txt')]

		# Only use n files (same as bboxes)
		if n:
			bboxes_ids = [file[:-11] for file in bboxes_files]
			res_files = [file for file in res_files if file[:-8] in bboxes_ids]

		# If files_to_use, only use specific files
		for i, file in enumerate(res_files):

			# Discard if ID not in files_to_use
			if files_to_use is not None and file[:-8] not in files_to_use:
				continue

			# Copy from source to target dir
			shutil.copyfile(source_dir+file, target_dir+file)


def create_enhanced_sg_representation(SOURCE_DIR, TARGET_DIR, functions=None, n_samples=None):
	"""Get enhanced and additional bounding boxes."""

	# Set bbox functions
	if functions is None:
		functions = [remove_duplicate_nodes, clean_labels, remove_multiple_representations, add_plural_objects, add_background_attribute, add_color_attribute]

	# Get files to use
	FILES_TO_USE = cfg.file_set
	with open(FILES_TO_USE) as infile:
		used_files = infile.read().split('\n')

	# Copy original representation files
	copy_files(SOURCE_DIR, TARGET_DIR, n=None, files_to_use=used_files, copy_res_files=True)

	# For each function
	for funct in functions:

		# Get bounding boxes files from target dir
		bboxes_files = [file for file in os.listdir(TARGET_DIR) if file.endswith('bboxes.txt')]

		# Get file ID
		for i,file in enumerate(bboxes_files):
			file_ID = re.search(r'(\d+)', file).group(1)

			# Apply function
			funct(file_ID, TARGET_DIR)

	# Save reductions
	if overall_reduction_rate:
		rate = sum(overall_reduction_rate)/len(overall_reduction_rate)

	# Get phrase grounding performance
	vis_flickr_img_paths, vis_flickr_sentences, vis_sg_rels, performance, performance_lin_sum = evaluate.run_alignment(verbose=False)


def para_screening(SOURCE_DIR, TARGET_DIR, functions=None, n_samples=None, merge_bboxes=False, merge_labels=False):
	"""Parameter screening for bbounding box reduction."""

	# Set parameter steps
	parameters = [1.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

	# Set default parametersW
	iou_threshold_o = 0.5
	neighbor_threshold_o = 0.5
	smallerbox_p = 1.1
	smallerbox_o = 1.1

	# Best settings
	# ---------------
	# iou_threshold_p = 0.8
	# neighbor_threshold_p = 0.3
	# smallerbox_p = 1.1
	#
	# iou_threshold_o = 0.4
	# neighbor_threshold_o = 0.3
	# smallerbox_o = 1.1

	# Set functions
	if functions is None:
		functions = [remove_duplicate_nodes, clean_labels, remove_multiple_representations, add_plural_objects, add_background_attribute, add_color_attribute]

	# Set outfile and which files to use
	PARA_SCREENING_OUTFILE = 'para_screening.csv'
	FILES_TO_USE = cfg.file_set

	# Get files to use
	with open(FILES_TO_USE) as infile:
		used_files = infile.read().split('\n')

	# Create and write headers to para screening outfile
	with open(PARA_SCREENING_OUTFILE,'w') as outfile:
		outfile.write('iou p,iou o,neighborhood overlap p,neighborhood overlap o,smallerBox iou p,smallerBox iou o,merge bboxes,merge labels,Reduction,People Reduction,Object Reduction,Aligner Performance,Linear Sum Performance,Distances,Abs Distance,Accuracy,Precision,Recall,F-Measure,Processing Time'+'\n')

	# For each parameter setting
	for iou_threshold_p in parameters:

		for neighbor_threshold_p in parameters:

					# Get start time
					start_time = timeit.default_timer()

					# Copy files to target dir
					copy_files(SOURCE_DIR, TARGET_DIR, n=None, files_to_use=used_files, copy_res_files=True)

					# Write parameters out
					with open(PARA_SCREENING_OUTFILE,'a') as outfile:
						out_data = ",".join([str(iou_threshold_p), str(iou_threshold_o), str(neighbor_threshold_p), str(neighbor_threshold_o), str(smallerbox_p), str(smallerbox_o), str(merge_bboxes), str(merge_labels)])
						outfile.write(out_data+',')

					# For each function
					for funct in functions:

						# Get files to process
						bboxes_files = [file for file in os.listdir(TARGET_DIR) if file.endswith('bboxes.txt')]

						# For each file
						for i,file in enumerate(bboxes_files):

							# Get file ID and discard if not in files_to_use
							file_ID = re.search(r'(\d+)', file).group(1)

							# Skip if not relevant file
							if used_files is not None and file_ID not in used_files:
							   continue

							# If bbox reduction function, set parameters
							if funct.__name__ == "remove_multiple_representations":
								funct(file_ID, TARGET_DIR, iou_threshold_p, iou_threshold_o,
										neighbor_threshold_p, neighbor_threshold_o,
										smallerbox_p, smallerbox_o,
										merge_bboxes=merge_bboxes, merge_labels=merge_labels)

							# Otherwise, simply call function
							else:
								funct(file_ID, TARGET_DIR)


					# Make Reductions global
					global overall_reduction_rate
					global overall_people_reduction_rate
					global overall_object_reduction_rate

					# Compute reductions
					if overall_reduction_rate:
						red_rate = sum(overall_reduction_rate)/len(overall_reduction_rate)
						person_red_rate = sum(overall_people_reduction_rate)/len(overall_people_reduction_rate)
						object_red_rate = sum(overall_object_reduction_rate)/len(overall_object_reduction_rate)
					else:
						red_rate = -1
						person_red_rate = -1
						object_red_rate = -1

					# Get phrase grounding performance
					vis_flickr_img_paths, vis_flickr_sentences, vis_sg_rels, performance, performance_lin_sum = evaluate.run_alignment(verbose=False)

					# Get distances and accuracies (evaluation)
					mean_distances, abs_distance, mean_accuracies, mean_precisions, mean_recalls, mean_f = gold_comp.object_reduction_evaluation(TARGET_DIR, FILES_TO_USE)

					# Get end time and save processing time
					end_time = timeit.default_timer()
					processing_time = end_time-start_time

					# Save results
					with open(PARA_SCREENING_OUTFILE,'a') as outfile:
						outfile.write(str(round(red_rate,4))+','+str(round(person_red_rate,4))+','+str(round(object_red_rate,4))+','+str(performance)+','+str(performance_lin_sum)+','+str(round(mean_distances,4))+','+str(round(abs_distance,4))+','+str(round(mean_accuracies,4))+','+str(round(mean_precisions,4))+\
						','+str(round(mean_recalls,4))+','+str(round(mean_f,4))+','+str(round(processing_time,4))+'\n')

					# Reset reduction rates
					overall_reduction_rate = []
					overall_people_reduction_rate = []
					overall_object_reduction_rate = []

def main():

	# Set paths
	SOURCE_DIR = cfg.SCENE_GRAPH_FLICKR_DIR + "/qualitative_sg_out_flickr_all/"
	TARGET_DIR = cfg.NEW_SG_REPR_DIR + "Flickr_original_noclean_attr_10000/"

	para_screening = False
	
	# SG_attr+plurals+red
	enhance_functions = [remove_duplicate_nodes, clean_labels, remove_multiple_representations, add_plural_objects, add_background_attribute, add_color_attribute]
	
	# SG attr+plurals
	enhance_functions = [remove_duplicate_nodes, clean_labels, add_plural_objects, add_background_attribute, add_color_attribute]
	
	# SG original+attr (no label cleaning)
	enhance_functions = [remove_duplicate_nodes, add_background_attribute, add_color_attribute]

	if para_screening:
		# Hyper-parameter screening
		para_screening(SOURCE_DIR, TARGET_DIR, functions=enhance_functions, merge_bboxes=True, merge_labels=True)

	else:
		# Generate new scene graph representation
		create_enhanced_sg_representation(SOURCE_DIR, TARGET_DIR, functions=enhance_functions)

		# Get performance
		_,_,_,_, performance = evaluate.run_alignment(verbose=False)
		print('Accuracy:', str(performance))


if __name__ == "__main__":
	main()
