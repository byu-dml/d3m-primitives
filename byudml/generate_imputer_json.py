import json

from imputer import RandomSamplingImputer

outfile = 'jsons/imputer_primitive.json'
with open(outfile, 'w') as f:
	primitive_json = RandomSamplingImputer(hyperparams=None).metadata.to_json()
	f.write(json.dumps(primitive_json, indent=4, sort_keys=True))

