# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# aligner_enhancements.py
# ----------------
# Aligner enhancements/extensions for number distinction (sign/plural filtering)
# and people assistance (candidate label and background tag filtering).

import config as cfg


# People trigger words
people_trigger_words = ['girl','boy','child','man','woman','person','guy','lady',
					'men','women','children','girls','boys','guys','ladies',
					'people','male','female','males','females','someone','youths','youth','teen','teens',
					'toddler','infant','toddlers','infants','baby','babies','bride','groom','daughter','adult','son','mother','father',
					'little girl','little boy', 'young girl','young boy','small girl', 'small boy',
										'worker','elderly women','elderly woman','old man','old woman','old lady',
										'middle-aged woman', 'middle-aged man', 'older man', 'older woman']

# Person candidate label dict
person_dict = {'boy':['boy','child','man','person','kid'],
			   'boys':['boy','child','man','men','person','kid','male','people','children'],
			   'guy':['boy','man','person'],
			   'guys':['boy','man','person','male','people','men'],
			   'girl':['girl','child','person','woman','kid',],
			   'girls':['girl','child','person','woman','kid','children','people','female'],
			   'child':['child','person','boy','girl','kid'],
			   'children':['child','person','boy','girl','kid','people','children'],
			   'kid':['child','person','boy','girl','kid'],
			   'kids':['child','person','boy','girl','kid','people','children'],
			   'male':['boy','man','guy','person'],
			   'female':['girl','woman','lady','person'],
			   'youth':['girl','boy','man','woman','person'],
			   'youths':['girl','person','woman','people','female','boy','man','male','men'],
			   'teen':['girl','boy','man','woman','person'],
			   'teens':['girl','person','woman','people','female','boy','man','male','men'],

			   'daughter':['girl','kid','child'],
			   'son':['boy','kid','child'],
			   'little girl':['girl','kid','child'],
			   'young girl':['girl','kid','child'],
			   'small girl':['girl','kid','child'],
			   'little boy':['boy','kid','child'],
			   'young boy':['boy','kid','child'],
			   'small boy':['boy','kid','child'],

			   'toddler':['child','person','boy','girl','kid'],
			   'toddlers':['child','person','boy','girl','kid','people','children'],
			   'infant':['child','person','boy','girl','kid'],
			   'infants':['child','person','boy','girl','kid','people','children'],
			   'baby':['child','person','boy','girl','kid'],
			   'babies':['child','person','boy','girl','kid','people','children'],

			   'man':['man','person','guy','boy'],
			   'worker':['man','person','guy','boy'],
			   'men':['man','person','guy','boy','male','people','men'],
			   'males':['man','person','guy','boy','male','people','men'],
			   'workers':['man','person','guy','boy','male','people','men'],
			   'woman':['woman','person','girl','lady'],
			   'women':['woman','person','girl','lady','female','people'],
			   'lady':['woman','person','lady','girl'],
			   'ladies':['woman','person','lady','girl','female','people'],
			   'females':['woman','person','girl','lady','female','people'],

			   'elderly man':['man','person'],
			   'older man':['man','person'],
			   'old man':['man','person'],
			   'middle-aged man':['man','person'],
			   'elderly woman':['woman','person','lady'],
			   'older woman':['woman','person','lady'],
			   'old woman':['woman','person','lady'],
			   'old lady':['woman','person','lady'],
			   'middle-aged woman':['woman','person','lady'],

			   'elderly men':['man','person','people','male','men'],
			   'older men':['man','person','people','male','men'],
			   'old men':['man','person','people','male','men'],
			   'middle-aged men':['man','person','people','male','men'],
			   'elderly women':['woman','person','lady','people','female'],
			   'older women':['woman','person','lady','people','female'],
			   'old women':['woman','person','lady','people','female'],
			   'middle-aged women':['woman','person','lady','people','female'],

			   'bride':['woman','person','girl','lady'],
			   'groom':['man','person','guy','boy'],
			   'mother':['woman','person','girl','lady'],
			   'father':['man','person','guy','boy'],
			   'parents':['people','person'],

			   'person':['woman','man','lady','guy','person','child','boy','girl','kid'],
			   'adult':['woman','man','lady','guy','person','boy','girl'],
			   'someone':['woman','man','lady','guy','person','child','boy','girl','kid'],
			   'people':['woman','man','lady','men','guy','person','child','boy','girl','kid','people','children','male','female'],
			   None: [],
}



def filter_for_number(sg_objects, plural_phrase):
	"""Filter out candidates with inadequate linugistic number (numerus: sing/plural)."""

	# Get all indices for plural objects
	plural_indices = [ind for ind, cat in enumerate(sg_objects[0]) if cat.endswith('-pl')]
	
	# Get plural categories
	plural_cats = [cat for cat in sg_objects[0] if cat.endswith('-pl')]
	
	# Initialize
	reduced_sg_objects = [[],[],[]]
	discard_indices = []

	# If phrase is plural phrase, reduce set of possible candidates
	if plural_phrase:
		
		# For each candidate object
		for j in range(len(sg_objects[0])):
			
			# If candidate object is plural, keep it
			if j in plural_indices:
				reduced_sg_objects[0].append(sg_objects[0][j])
				reduced_sg_objects[1].append(sg_objects[1][j])
				reduced_sg_objects[2].append(sg_objects[2][j])
			
			# If candidate object is singular, discard it
			else:
				discard_indices.append(j)

	# If phrase is singular phrase, reduce set of possible candidates
	else:
		
		# For each candidate object
		for j in range(len(sg_objects[0])):

			# If candidate object is singular, keep it
			if j not in plural_indices:
				reduced_sg_objects[0].append(sg_objects[0][j])
				reduced_sg_objects[1].append(sg_objects[1][j])
				reduced_sg_objects[2].append(sg_objects[2][j])

			# If candidate object is plural, discard it
			else:
				discard_indices.append(j)

	# Save reduction rate
	REDUCTION_RATE = (len(sg_objects[0])-len(reduced_sg_objects[0]), len(sg_objects[0]))

	return discard_indices, reduced_sg_objects


def filter_for_people_objects(objects, phrase, background_nodes):
	"""Filter out candidates for which label or background tag do not fit."""

	# Set default person tag
	person_tag = None
	
	# Get lower-cased phrase
	lower_phrase = phrase.lower()

	# If candidates are to be filtered by label
	if cfg.filter_candidate_list:
		
		# Check for each person word, whether phrase contains word
		for person_word in ['girl','boy','child','man','woman','person','guy','lady',
							'men','women','children','girls','boys','guys','ladies',
							'people','male','female','males','females','someone','youths','youth','teen','teens',
							'toddler','infant','toddlers','infants','baby','babies','bride','groom','daughter','adult','son','mother','father']:

			# Save as person tag (if several, save last on list)
			if person_word in lower_phrase.split():
				person_tag = person_word

		# Check for each person expression, whether phrase contains expression
		for person_exp in ['little girl','little boy', 'young girl','young boy','small girl', 'small boy',
							'worker','elderly women','elderly woman','old man','old woman','old lady',
							'middle-aged woman', 'middle-aged man', 'older man', 'older woman']:
	
			# Save as person tag (if several, save last on list)
			if person_exp in lower_phrase:
				person_tag = person_exp

	# Get background and foreground object indices
	background_indices = [ind for ind, cat in enumerate(objects[0]) if cat in background_nodes]
	foreground_indices = [ind for ind, cat in enumerate(objects[0]) if cat not in background_nodes]

	# If background filter is not applied, consider all objects as foreground
	if not cfg.filter_background:
		foreground_indices = [ind for ind, cat in enumerate(objects[0])]

	# Initialize
	disc_people_indices = []
	reduced_sg_objects = []
	suitable_people_indices = []
	selected_objects = [[],[],[]]
	people_indices = []
	
	# For each object candidate, filter out by label and/or background tag
	# (depending on settings)
	for ind, label in enumerate(objects[0]):
		
		# Get category
		category = label.split('-')[0]
		
		# If candidates are to be filtered by label
		if cfg.filter_candidate_list:
		
			# Use person tag to filter
			if person_tag:
				
				# If catogory in list of possible candidate labels, consider suitable
				if category in person_dict[person_tag]:
					suitable_people_indices.append(ind)
					
		# If candidates are not to be filtered by label, consider all suitable
		else:
			suitable_people_indices.append(ind)

	# For all suitable objects (returned by label filtering), 
	# check whether they are in forground
	for ind in suitable_people_indices:
		
		# If in foreground, consider suitable
		if ind in foreground_indices:
			selected_objects[0].append(objects[0][ind])
			selected_objects[1].append(objects[1][ind])
			selected_objects[2].append(objects[2][ind])
			
			# Save index
			people_indices.append(ind)

	# If there are not suitable candidates found in foreground, consider background
	if not selected_objects[0]:
		
		# Save all objects
		for ind in suitable_people_indices:
			selected_objects[0].append(objects[0][ind])
			selected_objects[1].append(objects[1][ind])
			selected_objects[2].append(objects[2][ind])
			
			# Save index
			people_indices.append(ind)

	# If there are any suitable candidates, use this new reduced set
	if selected_objects[0]:
		reduced_sg_objects = selected_objects
		
		# Get indices for candidates that are to be discarded
		disc_people_indices = [index for index in range(len(objects[0])) if index not in people_indices]

	# If there are no suitable candidates, use all original candidate objects
	if not reduced_sg_objects:
		reduced_sg_ojects = objects

	return person_tag, reduced_sg_objects, disc_people_indices

