# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# concept_config.py
# ----------------
# Configuration file for ConceptNet scripts.


DATA_DIR = "/home/mitarb/parcalabescu/"
HOME_DIR = "/home/students/suter/"

VERSION = "FINAL/Flickr_attr_plurals_red_10000/"
BBOXES_DIR = HOME_DIR+"neural-motifs/"+VERSION

CONCEPTNET_DIR = "data/conceptnet_full_di_rel_red.gpickle"

FILES_TO_USE = "data/used_files_5000.txt"


# Multi-processing settings
MULTI_PHRASE_CNCPTS = True
MULTI_PRO_SUBGRAPHS = True
MULTI_PRO_PAGERANKS = True
MULTI_PRO_SHORTPATH = True
MULTI_PRO_ALTS = True

updated_shortest_path_function = True
CUTOFF = 10

object_cats = [
		"airplane", "animal", "arm", "bag", "banana", "basket", "beach", "bear", "bed", "bench",
		"bike", "bird", "board", "boat", "book", "boot", "bottle", "bowl", "box", "boy", "branch",
		"building", "bus", "cabinet", "cap", "car", "cat", "chair", "child", "clock", "coat", "counter",
		"cow", "cup", "curtain", "desk", "dog", "door", "drawer", "ear", "elephant", "engine", "eye",
		"face", "fence", "finger", "flag", "flower", "food", "fork", "fruit", "giraffe", "girl", "glass",
		"glove", "guy", "hair", "hand", "handle", "hat", "head", "helmet", "hill", "horse", "house",
		"jacket", "jean", "kid", "kite", "lady", "lamp", "laptop", "leaf", "leg", "letter", "light",
		"logo", "man", "motorcycle", "mountain", "mouth", "neck", "nose", "number", "orange", "pant",
		"paper", "paw", "people", "person", "phone", "pillow", "pizza", "plane", "plant", "plate", "player",
		"pole", "post", "pot", "racket", "railing", "rock", "roof", "room", "screen", "seat", "sheep",
		"shelf", "shirt", "shoe", "short", "sidewalk", "sign", "sink", "skateboard", "ski", "skier",
		"sneaker", "snow", "sock", "stand", "street", "surfboard", "table", "tail", "tie", "tile", "tire",
		"toilet", "towel", "tower", "track", "train", "tree", "truck", "trunk", "umbrella", "vase",
		"vegetable", "vehicle", "wave", "wheel", "window", "indshield", "wing", "wire", "woman", "zebra"
	]
