from pathlib import Path
import os
from byudml import __imputer_path__, __metafeature_path__
import sys
sys.path.append(".")
from submission.utils import get_new_d3m_path

byu_dir = get_new_d3m_path()

# validate primitive jsons
primitive_jsons = Path(byu_dir).glob('**/primitive.json')
for filename in primitive_jsons:
    print(filename)
    os.system("./run_validation.py {}".format(filename))  

# check pipelines and meta files
for filename in Path(byu_dir).glob('**/*.json'):
    if filename in primitive_jsons:
        continue
    print(filename)
    os.system("python3 -m d3m pipeline describe {}".format(filename)) 
    os.system("python3 -m d3m runtime -d /datasets fit-score -m {} -p {}".format(filename.replace(".json", ".meta", 1), filename))  
 
