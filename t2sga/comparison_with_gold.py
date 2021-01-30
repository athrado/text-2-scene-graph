# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# comparison_with_gold.py
# ----------------
# Compare detected objects to gold annotations and evaluate whether
# object reduction improves representation.
# Slightly flawed measure but suitable for parameter screening. 

# Imports
import util.graph_functions as graph_functions
import os


# Gold entitiy class for Person
class GoldPersonEntity:
	def __init__(self, data):

		# Load data
		self.data = data
		split_data = self.data.split('\t')

		self.file_ID = split_data[0]
		self.ent_ID = split_data[1]
		self.string = split_data[2]
		self.coordinates = split_data[3][1:-1]
		self.is_female = (split_data[4] == 'True')
		self.is_male = (split_data[5] == 'True')
		self.is_child = (split_data[6] == 'True')
		self.is_person = True

		self.coordinates = tuple([float(c) for c in self.coordinates.split(',')])

# Gold entity class for Object
class GoldObjectEntity:
	def __init__(self, data):

		# Load data
		self.data = data
		split_data = self.data.split('\t')

		self.file_ID = split_data[0]
		self.ent_ID = split_data[1]
		self.string = split_data[2]
		self.coordinates = split_data[3][1:-1]
		self.is_person = False

		self.coordinates = tuple([float(c) for c in self.coordinates.split(',')])

# Load gold data for people
with open('gold_coordinates_people_clean_500.csv','r') as infile:
	gold_entities = infile.read()
	gold_entities = gold_entities.split('\n')[1:]
	gold_people_entities = [GoldPersonEntity(ent) for ent in gold_entities if ent.strip() != '']

# Load gold data for objects
with open('gold_coordinates_objects_clean_500.csv','r') as infile:
	gold_entities = infile.read()
	gold_entities = gold_entities.split('\n')[1:]
	gold_object_entities = [GoldObjectEntity(ent) for ent in gold_entities if ent.strip() != '']


def compare_entities(gold_people_entities, gold_object_entities, file_ID, people_nodes, object_nodes):
	"""Compare Gold entity to found entity bboxes."""

	# Get people and object nodes
	sg_nodes = people_nodes + object_nodes

	# Get gold entities for given image ID
	gold_people_entities = [ent for ent in gold_people_entities if ent.file_ID == file_ID]
	gold_object_entities = [ent for ent in gold_object_entities if ent.file_ID == file_ID]

	# Get gold entities
	gold_entities = gold_people_entities + gold_object_entities
	
	# Set nodes and gold entities (either people, objects or both)
	sg_nodes = people_nodes
	gold_entities = gold_people_entities

	sg_nodes = object_nodes
	gold_entities = gold_object_entities

	sg_nodes = sg_nodes
	gold_entities = gold_entities

	# Initalize
	included_nodes = []

	correct = 0
	total = 0
	TP = 0
	FP = 0
	FN = 0

	# Find best potential node
	for ent in gold_entities:

		if ent.is_person:
			
			# If entity is male
			if ent.is_male:
				
				# If male and child
				if ent.is_child:
					pot_nodes = [(label, cat, coordinates) for (label, cat, coordinates) in people_nodes if cat in ['boy','child']]
					
				# If male and unknown age
				else:
					pot_nodes = [(label, cat, coordinates) for (label, cat, coordinates) in people_nodes if cat in ['man','guy','person']]

			# If entity is female
			elif ent.is_female:
				
				# If female and child
				if ent.is_child:
					pot_nodes = [(label, cat, coordinates) for (label, cat, coordinates) in people_nodes if cat in ['girl','child']]
				
				# If female and unknown age
				else:
					pot_nodes = [(label, cat, coordinates) for (label, cat, coordinates) in people_nodes if cat in ['woman','lady','girl','person']]

			# If unknown gender
			else:
				
				# If child
				if ent.is_child:
					pot_nodes = [(label, cat, coordinates) for (label, cat, coordinates) in people_nodes if cat in ['girl','boy','child']]

				# If unknown gender and age
				else:
					pot_nodes = [(label, cat, coordinates) for (label, cat, coordinates) in people_nodes if cat in ['woman','man','guy','lady','person']]


		# If not person, no filtering
		else:
			pot_nodes = [(label, cat, coordinates) for (label, cat, coordinates) in object_nodes]

		# Get IoU scores for candidates
		iou_scores = [graph_functions.iou(ent.coordinates, coordinates) for (label, cat, coordinates) in pot_nodes]

		has_match = False

		# Test for each score, whether it is greater than 0.5 (= match)
		for i,score in enumerate(iou_scores):
			
			if score >=0.5:
				has_match = True
				# Get false positives
				FP +=1
				included_nodes.append(pot_nodes[i][0])
		
		# Get highest IoU score and use as best guess
		if iou_scores:
			max_iou_score = max(iou_scores)
			index = iou_scores.index(max_iou_score)

			coordinates = pot_nodes[index][1]
			best_guess = pot_nodes[index]
			
			# Count correct guesses
			if max_iou_score >= 0.5:
				correct += 1

		# Evaluation counters
		total += 1

		if has_match:
			TP += 1
			FP -= 1
		else:
			FN += 1

	# Get metrics
	if gold_entities:

		# Get accuracy
		accuracy = correct/total*100

		# How many gold nodes and SG nodes
		n_sg_nodes = len([n for n in sg_nodes if n[0] in included_nodes])
		n_superflous_sg_people = len([n for n in people_nodes if n[0] in excluded_people_nodes and n[0] not in included_people_nodes])
		n_gold_nodes = len(gold_entities)

		# Get distance and fraction
		distance = abs(n_sg_nodes-n_gold_nodes)
		fraction = n_gold_nodes/distance*100

		# Get distance (with corrected sign - absolute value)
		if n_gold_nodes > n_sg_nodes:
			sign_distance = -distance
		else:
			sign_distance = distance

		# Print counters
		print('TP', TP)
		print('FP', FP)
		print('FN', FN)
		print('Gold', n_gold_nodes)

		# Compute precision
		if (TP+FP) == 0:
			precision = 0
		else:
			precision = TP / (TP+FP)
			
		# Compute recall
		recall = TP / (TP+FN)
		
		# Compute F1-score
		if precision+recall == 0:
			f_measure = 0
		else:
			f_measure = (precision*recall)/(precision+recall)*2

		return fraction, sign_distance, accuracy, precision, recall, f_measure

	# If no matches found
	else:
		return (None, None, None, None, None, None)



def object_reduction_evaluation(SOURCE_DIR, used_files):
	"""Evaluate object reduction for removing multiple representations."""


	# Which files to use
	with open(used_files) as infile:
		used_files = infile.read().split('\n')
		used_files = [file for file in used_files if file.strip() != '']

	# Load gold people entities
	with open('gold_coordinates_people_clean_500.csv','r') as infile:
		gold_entities = infile.read()
		gold_entities = gold_entities.split('\n')[1:]
		gold_people_entities = [GoldPersonEntity(ent) for ent in gold_entities if ent.strip() != '']

	# Load gold object entities
	with open('gold_coordinates_objects_clean_500.csv','r') as infile:
		gold_entities = infile.read()
		gold_entities = gold_entities.split('\n')[1:]
		gold_object_entities = [GoldObjectEntity(ent) for ent in gold_entities if ent.strip() != '']

	# Initialize
	all_distances = []
	all_accuracies = []
	all_abs_distances = []
	all_precisions = []
	all_recalls = []
	all_f = []

	# For each image
	for file_ID in used_files:

		# Get all nodes
		with open(SOURCE_DIR+file_ID+'_bboxes.txt','r') as infile:
			data = infile.read()
			nodes = data.split('\n')
			nodes = [n for n in nodes if n.strip() != '']

		# Initialize
		people_tags = ['person','man','woman','girl','boy','child','kid','lady','guy']

		people_nodes = []
		object_nodes = []

		# Process each node
		for node in nodes:
			
			# Get label
			node_label, coordinates = node.split(';')
			node_label = node_label.strip()

			# Skip plurals
			if node_label.endswith('pl'):
				continue
			   
			# Get category and coordinates 
			node_category = node_label.split('-')[0]
			coordinates = coordinates.strip()[1:-1]
			coordinates = tuple([float(c) for c in coordinates.split(',')])

			# Save as people or object node
			if node_category in people_tags:
				people_nodes.append((node_label, node_category, coordinates))

			else:
				object_nodes.append((node_label, node_category, coordinates))

		# Get distance, accuracy. precision, recall, F-score
		distance, abs_distance, accuracy, precision, recall, f_measure = compare_entities(gold_people_entities, gold_object_entities, file_ID, people_nodes, object_nodes)

		# Save metrics
		if distance != None:
			all_distances.append(distance)
			all_abs_distances.append(abs_distance)
			all_accuracies.append(accuracy)
			all_precisions.append(precision)
			all_recalls.append(recall)
			all_f.append(f_measure)

	# Compute metrics
	mean_distances = sum(all_distances)/len(all_distances)
	abs_distance = sum(all_abs_distances)
	mean_accuracies = sum(all_accuracies)/len(all_accuracies)
	mean_precisions = sum(all_precisions)/len(all_precisions)
	mean_recalls = sum(all_recalls)/len(all_recalls)
	mean_f = sum(all_f)/len(all_f)
		
	# Return
	return (mean_distances, abs_distance, mean_accuracies, mean_precisions, mean_recalls, mean_f)


def main():
	
	# Set source directory
	SOURCE_DIR = "../neural-motifs/enhanced_Flickr_red_test/"
	
	# Get metrics
	mean_distances, abs_distance, mean_accuracies, mean_precisions, mean_recalls, mean_f = bbox_reduction_evaluation(SOURCE_DIR, 'used_files/used_files_200.txt')

	# Print metrics
	print(mean_distances)
	print(abs_distance)
	print(mean_accuracies)
	print()
	print('Distances:', mean_distances)
	print('Abs distances:', abs_distance)
	print('Accuracies:', mean_accuracies)
	

if __name__ == "__main__":
	main()
