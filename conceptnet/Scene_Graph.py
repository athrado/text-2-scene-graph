# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# Scene_Graph.py
# ----------------
# Classes/data management for scene graphs, object nodes 
# and phrase entities and concepts and their attributes.


# Imports
import re
import os
import sys
from shutil import copyfile
import collections

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as fm

from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET

sys.path.insert(1, '../t2sga/')
# Import Scene Graph modules
import config as cfg
import util.graph_functions as graph_funct
import box_geometry as box



# Scene graph labels (150) divided into Flickr30k categories

bodypart_set = ['hair','face','hand','ear','head','eye','mouth','arm','finger',
				'foot','head','leg','neck','nose','paw']
clothes_set = ['shirt','pant','hat','jacket','shoe','jean','glove','short',
			   'cap','coat','helmet','sneaker','sock']
person_nodes = ['woman','man','girl','boy','person','child','kid','lady','guy']
animal_nodes = ['animal','dog','cat','zebra','bear','bird','giraffe','elephant',
				'horse','sheep','cow']
vehicle_nodes = ['airplane','bike','boat','car','truck','bus','motorcycle',
				 'plane','truck','vehicle']


# Scene graph relations
relations = ["on", "has", "near", "behind", "of", "wearing", "in", "made of",
			"holding", "with", "under", "in front of", "sitting on", "at",
			"above", "watching", "walking on", "riding", "standing on",
			"over","attached to", "carrying", "covered in", "looking at",
			"for", "hanging from", "eating", "laying on", "and", "wears",
			"belonging to", "using", "between", "along", "covering", "parked on",
			"part of", "on back of", "against", "lying on", "painted on",
			"to", "from"]



def get_ent_category(cat):
	"""Get Flickr30k category for Scene Graph label."""

	if cat in bodypart_set:
		return 'bodyparts'
	elif cat in clothes_set:
		return 'clothing'
	elif cat in person_nodes:
		return 'people'
	elif cat in animal_nodes:
		return 'animals'
	elif cat in vehicle_nodes:
		return 'vehicles'
	else:
		return 'other'


class Node:
	"""Class for Scene Graph Nodes"""

	def __init__(self, label, coordinates, center, rank, SceneGraph):

		# Get label, category and ID
		self.label = label
		self.category, self.ID = label.split('-')

		# Flickr30k Entity category corresponding to label
		self.ent_category = get_ent_category(self.category)

		# Rank by ID (not used)
		self.rank = int(rank)
		if self.rank == 'pl':
			rank = 10000

		# Get coordinates and area size
		self.coordinates = coordinates
		self.center = center
		self.xmin, self.ymin, self.xmax, self.ymax = coordinates
		self.area = graph_funct.get_box_size(self.coordinates)

		# Initialize neighbors and edges dicts/lists
		self.neighbors = []
		self.out_edges = {}
		self.in_edges = {}
		self.node_triples = []

		# For each triple
		for n1, rel, n2 in SceneGraph.all_triples:

			if n1 == label or n2 == label:
				self.node_triples.append((n1, rel, n2))

			# If n1 is current node, save as out edge
			if n1 == label:
				if rel in self.out_edges.keys():
					self.out_edges[rel].append(n2)
				else:
					self.out_edges[rel] = [n2]
				self.neighbors.append(n2)

			# If n2 is current node, save as in edge
			elif n2 == label:
				if rel in self.in_edges.keys():
					self.in_edges[rel].append(n1)
				else:
					self.in_edges[rel] = [n1]
				self.neighbors.append(n1)


		# Set bodypart and clothes lists
		bodypart_set = ['hair','face','hand','ear','head','eye','mouth','arm','finger','foot','head','leg','neck','nose']
		clothes_set = ['shirt','pants','hat','jacket','shoe','jeans','glove','shorts','cap','coat','helmet','sneaker','sock','trunks','trousers']

		self.bodyparts = []
		self.clothes = []

		# If node is person node, save "has" and "wearing" relations and nodes
		if self.category in ['woman','man','girl','boy','person','child','kid','lady','guy']:

			if 'has' in self.out_edges.keys():
				self.bodyparts = sorted([n for n in self.out_edges['has'] if n.split('-')[0] in bodypart_set])
			if 'wearing' in self.out_edges.keys():
				self.clothes = sorted([n for n in self.out_edges['wearing'] if n.split('-')[0] in clothes_set])

	def __str__(self):
		"""Print label."""
		return self.label

class SceneGraph:
	"""Class for Scene Graph."""

	def __init__(self, file_ID, is_flickr=True, flickr_version='all', sg_data_dir=None):

		self.file_ID = file_ID

		# Set paths (if Flickr data)
		if is_flickr:
			self.path = cfg.SCENE_GRAPH_FLICKR_DIR + "/qualitative_sg_out_flickr_" + flickr_version + "/"
			self.bboxes_file = self.path + self.file_ID + '_bboxes.txt'
			self.res_file = self.path + self.file_ID + '_res.txt'
			self.original_image = cfg.FLICKR_IMAGES + file_ID + '.jpg'

		# Set directories by arguments
		if sg_data_dir:
			self.bboxes_file = sg_data_dir + self.file_ID + '_bboxes.txt'
			self.res_file = sg_data_dir + self.file_ID + '_res.txt'

		# Initialize
		self.all_triples = []
		self.all_nodes = []
		self.all_edges = []
		self.all_edge_labels = {}
		self.all_relations = []

		self.node_centers = {}
		self.node_coordinates = []

		self.label_dict = {}

		self.nodes_dict = {}
		self.nodes = []
		self.triples = []


	def get_triples(self):
		"""Get all triples for Scene Graph/image."""

		# Open file
		with open(self.res_file) as infile:
			data = infile.read()

		# Prepare data
		data = data.strip()
		data = data.split('\n')

		# For each line, get nodes and relation
		for i, line in enumerate(data):
			if line.strip() == '':
				continue

			# Get nodes and relation
			elements = line.split(' - ')
			node_1 = elements[0].strip()
			relation = elements[1].strip()
			node_2 = elements[2].strip()

			# Save nodes
			self.all_nodes.append(node_1)
			self.all_nodes.append(node_2)

			# Save edge and edge label (relation)
			self.all_edges.append((node_1, node_2))
			self.all_edge_labels[(node_1, node_2)] = relation
			self.all_relations.append(relation)

			# Save triples
			self.all_triples.append((node_1, relation, node_2))

		# Set nodes
		self.all_nodes = list(set(self.all_nodes))

		# Get node types = labels
		self.node_types = [node.split('-')[0] for node in self.all_nodes]

	def get_bboxes(self):
		"""Get all bounding boxes for Scene Graph/image."""

		node_coordinates = []

		# Open corresponding bounding box file by ID
		with open(self.bboxes_file) as infile:

			# Prepare data
			data = infile.read().strip()
			node_coordinates = data.split('\n')
			node_coordinates = [n for n in node_coordinates if n.strip() != '']

			# For each node
			for node in node_coordinates:

				# Save and clean name and coordinates
				node_name, coordinates = node.split(';')
				node_name = node_name.strip()
				coordinates = coordinates.strip()[1:-1]

				# Get boundary coordinates as floats
				xmin, ymin, xmax, ymax = coordinates.split(', ')
				xmin = float(xmin)
				xmax = float(xmax)
				ymin = float(ymin)
				ymax = float(ymax)

				coordinates = (xmin, ymin, xmax, ymax)

				# Compute center position for x and y
				x_position = xmin + (xmax - xmin)/2
				y_position = ymin + (ymax - ymin)/2

				# Get center
				center = (x_position, y_position)

				# Save node center and coordinates
				self.node_centers[node_name] = center
				self.node_coordinates.append((node_name, coordinates, center))

	def get_nodes(self):
		"""Get all nodes and their information for Scene Graph/image."""

		# Get coordinates
		if not self.node_coordinates:
			self.get_bboxes()

		# Get triples
		if not self.all_triples:
			self.get_triples()

		processed_nodes = []

		# Process each node
		for i, (node_name, coordinates, center) in enumerate(self.node_coordinates):

			# Skip if already processed
			if node_name in processed_nodes:
				continue

			processed_nodes.append(node_name)

			# Get node as Node class
			new_node = Node(node_name, coordinates, center, i, self)

			# Save Node
			self.nodes.append(new_node)
			self.nodes_dict[node_name] = new_node


		# Save all triples
		for n1, rel, n2 in self.all_triples:
			try:
				self.triples = (self.nodes_dict[n1], rel, self.nodes_dict[n2])
			except KeyError:
				continue


	def get_label_dict(self):
		"""Get category/label dict for Scene Graph/image."""

		# Set node sets
		person_nodes = ['woman','man','girl','boy','person','child','kid','lady','guy']

		similar_object_nodes = ['truck','car','motorcycle','bike','book','paper','glass','cup',
								'table','desk','stand']

		# Get nodes
		if not self.nodes:
			self.get_nodes()

		# Initialize label dict
		self.label_dict = {}
		self.label_dict['person_'] = []
		self.label_dict['car_'] = []
		self.label_dict['bike_'] = []
		self.label_dict['book_'] = []
		self.label_dict['glass_'] = []
		self.label_dict['table_'] = []

		# For each node, check whether they are part of label set
		for node in self.nodes:

			# Save person nodes with special person_ key
			if node.category in person_nodes:
				self.label_dict['person_'].append(node)

			if node.category in ['truck','car']:
				self.label_dict['car_'].append(node)

			if node.category in ['motorcycle','bike']:
				self.label_dict['bike_'].append(node)

			if node.category in ['book','paper']:
				self.label_dict['book_'].append(node)

			if node.category in ['glass','cup']:
				self.label_dict['glass_'].append(node)

			if node.category in ['table','desk','stand']:
				self.label_dict['table_'].append(node)

			# Save node and label in dict
			if node.category in self.label_dict.keys():
				self.label_dict[node.category].append(node)
			else:
				self.label_dict[node.category] = [node]

		return self.label_dict


	def get_image_ratio(self):
		"""Get image ratio."""

		# Get original image data
		image_file = self.original_image
		img = mpimg.imread(image_file,0)

		# Compute ratio of original image to downsampled image
		max_axis = max(img.shape[:2])
		self.ratio = max_axis/1024
		self.area = img.shape[0]*img.shape[1]
		return self.ratio, self.area


	def overlay_on_image(self, image_file_path, outpath='', format="jpg", save_original=True):
		"""Draw Scene Graph and lay over original image."""

		# Set output filename
		filename = 'SG_' + self.file_ID + '.png'

		# Get original image path
		image_file = image_file_path + self.file_ID + "."+format

		if save_original:
			# Copy original file in output folder for comparison
			copyfile(image_file, outpath+'SG_'+self.file_ID+'_original.jpg')


		# Plot original image
		img = mpimg.imread(image_file,0)
		# plt.imshow(img)
		ax = plt.gca()

		# Compute ratio of original image to downsampled image (1024, 768, 3)
		max_axis = max(img.shape[:2])
		self.ratio = max_axis/1024

		# Create Graph
		G=nx.Graph()

		# Add edges
		G.add_edges_from(self.all_edges)

		# Apply ratio to node positions in order to adjust to image
		self.node_centers = {node:(pos[0]*self.ratio, pos[1]*self.ratio) for node,pos in self.node_centers.items()}

		# Add positions
		pos = self.node_centers

		# Draw
		nx.draw(G,pos,edge_color='black',width=0.5,linewidths=0.8,\
		node_size=1250,node_color='white',shape='o', node_angle=0, font_size=8, font_color='black',alpha=0.6,\
		labels={node:node for node in G.nodes()}, ax=ax)

		# Draw labels
		nx.draw_networkx_edge_labels(G,pos,edge_labels=self.all_edge_labels,font_color='red', font_size=8)

		# Clean and save
		plt.axis('off')
		plt.tight_layout()
		plt.savefig(outpath+filename, format="PNG")

		# Close
		plt.close()

		self.nodes_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
		self.degree_dict = {k:v for k,v in G.degree}

	def save_captions_as_image(self, dataset, outpath=''):
		"""Save captions as image."""

		# Get captions by file ID
		captions = dataset.captions_dict[self.file_ID]

		# Join to string
		captions_to_print = "\n".join(captions)

		# Load font (arial)
		font = ImageFont.truetype("/home/jsuter/repositories/arial.ttf", 15)

		# Set image settings
		img = Image.new('RGB', (600, 150), color = (73, 109, 137))
		d = ImageDraw.Draw(img)
		d.text((10,10), captions_to_print, fill=(255,255,0), font=font)

		# Save image
		img.save(outpath+'SG_'+self.file_ID+'_captions.png')


	def create_network(self, outpath='', format="jpg"):
		"""Draw Scene Graph and lay over original image."""

		# Get original image path
		image_file = self.original_image

		# Plot original image
		img = mpimg.imread(image_file,0)
		plt.imshow(img)
		ax = plt.gca()

		# Compute ratio of original image to downsampled image (1024, 768, 3)
		max_axis = max(img.shape[:2])
		self.ratio = max_axis/1024
		ratio = self.ratio

		# Create Graph
		G=nx.Graph()

		# Add edges
		G.add_edges_from(self.all_edges, label=self.all_edge_labels)

		# Apply ratio to node positions in order to adjust to image
		self.node_centers = {node:(pos[0]*ratio, pos[1]*ratio) for node,pos in self.node_centers.items()}

		# Add positions
		pos = self.node_centers

		# Draw
		nx.draw(G,pos,edge_color='black',width=0.5,linewidths=0.8,\
		node_size=1250,node_color='white',shape='o', node_angle=0, font_size=8, font_color='black',alpha=0.6,\
		labels={node:node for node in G.nodes()}, ax=ax)

		# Draw labels
		#nx.draw_networkx_edge_labels(G,pos,edge_labels=self.all_edge_labels,font_color='red', font_size=8)

		# Clean and save
		plt.axis('off')
		plt.tight_layout()
		plt.show()
		#plt.savefig(outpath+filename, format="PNG")

		# Close
		plt.close()

		# Save degrees
		self.nodes_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
		self.degree_dict = {k:v for k,v in G.degree}


class Entity():
	"""Class for entities in captions."""

	def __init__(self, data):

		# Match parts of entity string
		matches = re.search(r'EN#(.*?)\/(.*?)\s(.*)', data)

		# Save full data string
		self.data = data

		# Get ID
		self.ID = matches.group(1)

		# Get category
		self.category = matches.group(2)

		# Get string
		self.string = matches.group(3).lower()

		# Get object (final word in string)
		self.object = self.string.split(' ')[-1]

		# Is entity scene?
		self.is_scene = False

		# Initialize coordinates
		self.xmax = None
		self.xmin = None
		self.ymax = None
		self.ymin = None

		self.coordinates = None

		self.area = None


	def set_bbox(self, coordinates):
		"""Set bounding box coordinates."""

		if coordinates == None:
			self.is_scene = True
		else:
			self.xmin, self.ymin, self.xmax, self.ymax = coordinates
			self.coordinates = (self.xmin, self.ymin, self.xmax, self.ymax)
			self.area = graph_funct.get_box_size(self.coordinates)


class Dataset():
	"""Dataset class for loading Flickr datasets."""

	def __init__(self, path):

		# Set paths and file names
		self.path = path
		self.capation_path = self.path + "/flickr30k-captions/"
		self.captions_file = self.capation_path+"results_20130124.token"

		self.entities_sents_path = self.path + "/flickr30k_entities/Sentences/"
		self.entites_bboxes_path = self.path + "/flickr30k_entities/Annotations/"

		# Set dicts and lists
		self.captions_dict = {}
		self.entities_dict = {}
		self.all_entities = []


		self.captions_with_ent = {}
		self.word_ent_ID_dict = {}
		self.ent_ID_dict = {}

		self.caption_phrase_dict = {}
		self.all_words = []
		self.all_relations = []


	def get_captions(self):
		"""Get all captions."""

		# Open caption file
		with open(self.captions_file) as infile:
			data = infile.read()

		# Clean captions
		captions = data.split('\n')
		captions = [cap for cap in captions if cap.strip() != '']

		for cap in captions:
			for rel in relations:
				if rel in cap:
					self.all_relations.append(rel)

		# Get captions sets (5 each)
		captions_per_image = [captions[i:i+5] for i in range(0, len(captions), 5)]

		# For each caption set
		for caption_set in captions_per_image:

			# Get file
			file_ID  = caption_set[0].split('\t')[0]
			file_ID = file_ID[:-6]

			# Get captions
			extracted_captions = [cap.split('\t')[1] for cap in caption_set]

			# Save captions for file ID
			self.captions_dict[file_ID] = extracted_captions


	def get_entities(self):
		"""Get all entities."""

		# Get all files
		files = [f for f in os.listdir(self.entities_sents_path) if f.endswith('.txt')]

		# Process each file
		for file in files:

			# Get file ID
			file_ID = file[:-4]

			entity_dict = {}
			bbox_dict = {}

			# Open entity file and read
			with open(self.entities_sents_path+file) as infile:
				data = infile.read()
				data = data.strip()

			# Load bounding box file as XML
			root = ET.parse(self.entites_bboxes_path+file_ID+'.xml').getroot()

			for object_xml in root.findall('size'):
				width = int(object_xml.findall('width')[0].text)
				height = int(object_xml.findall('height')[0].text)

			# neural-motifs scaled such that the greatest dim is 1024
			if width > height:
				newWidth = 1024
				newHeight = newWidth * height / width
			else:
				newHeight = 1024
				newWidth = newHeight * width / height

			# For each object
			for xml_object in root.findall('object'):

				# Find names and scene
				names = xml_object.findall('name')
				scene = xml_object.find('scene')

				# Check whether object is scene
				is_scene = True if scene is not None else False

				# If not, load bounding box cordinates
				if not is_scene:

					ymin = int(xml_object.find('bndbox/ymin').text) * newWidth/width
					ymax = int(xml_object.find('bndbox/ymax').text) * newWidth/width
					xmax = int(xml_object.find('bndbox/xmax').text) * newHeight/height
					xmin = int(xml_object.find('bndbox/xmin').text) * newHeight/height

				# Save coordinates for object (None for scene or non-visual object)
				if is_scene:
					for name in names:
						bbox_dict[name.text]  = None
				else:
					for name in names:
						if name.text in bbox_dict.keys() and bbox_dict[name.text] !=0:
							bbox_dict[name.text].append((xmin, ymin, xmax, ymax))

						else:
							bbox_dict[name.text] = [(xmin, ymin, xmax, ymax)]

			# Merge coordinates for gold entity bounding boxes
			for key in bbox_dict.keys():
				all_coords = bbox_dict[key]
				if all_coords != None:
					merged_coords = (graph_funct.multiple_box_union(all_coords))
					bbox_dict[key] = merged_coords

			# Get captions
			captions = data.split('\n')

			# For each cap
			for i, cap in enumerate(captions):

				# Save caption for image
				self.captions_with_ent[(file_ID, str(i))] = cap

				word_ID_dict = {}
				cap_entities = []
				caption_phrases = []

				# Parse out phrases
				cap_raw = re.sub(r'(EN#\d+\/\w+.*?)(\s+)([\w\"\']+)','\g<1>$$$\g<3>',cap)
				cap_words_raw = cap_raw.split(' ')

				current_ID = None
				current_phrase = ''

				# Divide up caption into phrases and assign entity ID to each phrase word
				for word_id, word in enumerate(cap_words_raw):
					if word.startswith('['):
						
						# Get ID and word
						ID = re.search(r'EN#(\d+)\/.*', word).group(1)
						word = word.split('$$$')[-1]
						current_ID = str(ID)

						single = False

						# If entity = 0, don't save
						if current_ID == '0':
							current_ID = None

						# If end of phrase (single word phrase), save as caption phrase
						if word.endswith(']'):
							word = word[:-1]
							single = True
							current_phrase += word
							caption_phrases.append((current_phrase, current_ID))
							current_phrase = ''
						 
						# If not yet end of phrase, keep appending
						else:
							current_phrase += '_'+word
						
						# Save word ID and word
						word_ID_dict[(word_id, word)] = current_ID

						if single:
							current_ID = None
							single = False

					else:
						# If end of non-single word phrase
						if word.endswith(']'):
							single = False

							if current_ID == '0':
								current_ID = None

							if word.endswith(']'):
								word = word[:-1]
								single = True
	
							# Save phrase
							current_phrase += '_'+word
							caption_phrases.append((current_phrase, current_ID))
							current_phrase = ''

							# Save word ID and word
							word_ID_dict[(word_id, word)] = current_ID

							if single:
								current_ID = None
								single = False
	
						# Print word ID and word
						else:
							word_ID_dict[(word_id, word)] = current_ID
							current_phrase += '_'+word

				entity_words = {}
				ent_IDs = {}

				# Find and clean all entity strings
				entities = re.findall(r'\[\/.*?\]', cap)
				entities = [ent[1:-1] for ent in entities]

				# For each entity
				for ent in entities:

					# Entity instance
					ent = Entity(ent)

					# If ID does not indicate "non-visual"
					if ent.ID != '0':

						# Save bounding box
						ent.set_bbox(bbox_dict[ent.ID])

						# Append entities
						cap_entities.append(ent)
						self.all_entities.append(ent)

					# If there are coordinates, save entity for entity ID, otherwise None
					if ent.coordinates == None:
						ent_IDs[str(ent.ID)] = None
					else:
						ent_IDs[str(ent.ID)] = ent

				# Save caption entities
				entity_dict[i] = cap_entities

				# Save caption phrases and entity ID dicts
				self.caption_phrase_dict[(file_ID),str(i)] = caption_phrases
				self.word_ent_ID_dict[(file_ID, str(i))] = word_ID_dict
				self.ent_ID_dict[(file_ID, str(i))] = ent_IDs

			# Save entity by file ID
			self.entities_dict[file_ID] = entity_dict



class PhraseConcept:
	"""Class for Phrase Concepts used for ConceptNet extension."""

	def __init__(self, data, image_ID, caption_ID, concept_text, ent_ID, entity, is_plural):
		
		# Data
		self.line = data
		self.caption_ID = caption_ID
		self.image_ID = image_ID

		self.concept_text = concept_text
		self.ent_ID = ent_ID
		self.entity = entity
		self.is_plural = is_plural
