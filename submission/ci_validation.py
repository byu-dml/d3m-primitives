from pathlib import Path
import os
from byudml import __imputer_path__, __metafeature_path__
import sys
sys.path.append(".")
from submission.utils import get_new_d3m_path

byu_dir = get_new_d3m_path()
print("##### Validating json files with D3M Validation #####")

primitive_jsons = Path(byu_dir).glob('**/primitive.json')
# TODO: add validation for primitive jsons
# for filename in primitive_jsons:
#     # strip the init file path off, since we are changing directories
#     filename_real = "/".join(str(filename).split("/")[2:])
#     print("Validating primitive:", filename_real)
#     os.system("cd submission/primitives/; python3 run_validation.py {}; cd ../..".format(filename_real))  

# check pipelines and meta files
for filename in Path(byu_dir).glob('**/*.json'):
    if str(filename).split("/")[-1] == "primitive.json":
        # don't validate the primitive.json as a pipeline
        continue
    # strip the init file path off, since we are changing directories
    print("Validating pipeline:", filename)
    meta_name = str(filename).replace(".json", ".meta")
    os.system("python3 -m d3m pipeline describe {}".format(filename)) 
    os.system("python3 -m d3m runtime -d /datasets fit-score -m {} -p {}".format(meta_name, filename))  
 
