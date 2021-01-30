# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg


# config.py
# ----------------
# Configurations for phrase grounding and evaluation.


# Data paths

# Scene graph representations
SCENE_GRAPH_FLICKR_DIR = "/home/mitarb/parcalabescu/neural-motifs/"
NEW_SG_REPR_DIR = "/home/students/suter/neural-motifs/FINAL/"

# Flickr data
FLICKR_DATA = "/home/mitarb/parcalabescu/"
FLICKR_IMAGES = "/home/mitarb/parcalabescu/flickr30k-images/"


# Evaluation Parameters
N_IMAGES =  10000 # number of used images
REC_k = 1  # Recall @k. k=1 is evquivalent to the accuracy

# General settings
analysis = False
visualization = False
deprecated_version = False
original_version = False
baseline = False
comment = ''

# Basic alignment enhancements
exclude_failed_embeddings = False
fix_typos = False

# Alignment enhancements
include_colors = False
filter_people = False
filter_number = False

filter_background = False
filter_candidate_list = False

# Ablation
only_color_samples = False
only_people_samples = False
only_typo_samples = False

# Color settings
best_sec =  1500
best_diff = 10000000
color_n_to_use = None
kmeans = True

# Linear sum assignments
unique_assignments = False

prefer_plural = False

# ConceptNet enhancements
apply_filters = False
alt_sim_measures = False

# Alternatives /support concepts
alt_ver_tag = ['','_emb_version'][0]
alternative_path = ['_new_3',''][0]

include_alternatives = False
include_alt_objects = False
new_objects = False

include_alt_phrases = False
phrase_mean_cosim = False

acc_over_cosim = False # Accuracy rather than weighted sum from cosine sim (default:False)

number_of_alts = 1
number_of_alts_phrase = 1

phrase_min_acc = 50
object_min_acc = 50


# Scene Graph Representation version
#version = ['original_attr_10000', 'basic_attr_plurals_10000',
#		   'basic_attr_plurals_red_10000', 'para','test'][0]

version = ['original_attr_10000','attr_plurals_10000',
		   'attr_plurals_red_10000', 'original_noclean_attr_10000'][-1]

file_set = 'used_files/used_files_10000.txt'
#file_set = 'used_files/used_files_500_conceptnet.txt'


# Visualization
N_VIS = 20  # Number of images to visualise
VIS_COULD_HAVE_FOUND = False  # (EXPERIMENTAL)

# Merging
MERGE = True # true: merge multiple bboxes per phrase to a union bbox
MERGE_TYPE = "argwhere"  # num_flickr or argwhere

# Embedding settings (default)
EMBEDDINGS = 'word2vec'  # glove, bert, autoextend, word2vec
EMBED_DIM = 300  # embedding dimension

# Set dimensions
if EMBEDDINGS == 'word2vec':
	EMBED_DIM = 300
if EMBEDDINGS == 'glove':
	EMBED_DIM = 200



# Path to sentence annotations
FLICKR_SENTENCES = FLICKR_DATA + "flickr30k_entities/Sentences/*.txt"

# Patht to bounding box annotations
FLICKR_BOXES = FLICKR_DATA + "flickr30k_entities/Annotations/{}.xml"

# Path to result files (triples)
SG_OUT = SCENE_GRAPH_FLICKR_DIR+"/qualitative_sg_out_flickr_all/{}_res.txt"

# Set paths for scene graph representations and triples
SG_BOXES = NEW_SG_REPR_DIR+"/Flickr_"+version+"/{}_bboxes.txt"
SG_OUT = NEW_SG_REPR_DIR+"/Flickr_"+version+"/{}_res.txt"

# In deprecated version, use old data
if deprecated_version:
	SG_BOXES = SCENE_GRAPH_FLICKR_DIR+"/qualitative_sg_out_flickr_all/{}_bboxes.txt"
	SG_OUT = SCENE_GRAPH_FLICKR_DIR + "/qualitative_sg_out_flickr_all/{}_res.txt"

