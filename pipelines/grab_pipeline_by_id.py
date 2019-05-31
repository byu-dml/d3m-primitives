import json
import pymongo
from bson import json_util

real_mongo_port = 12345
lab_hostname = "machine"



mongo_client = pymongo.MongoClient(lab_hostname, real_mongo_port)

pipelines_to_grab = ["0e5b9a6e-df51-4486-914a-8d862431b81d"]


for pipeline_id in pipelines_to_grab:
    collection = mongo_client.metalearning.pipelines
    pipeline_to_write = collection.find({"id": pipeline_id})
    for pipeline in pipeline_to_write:
        print(pipeline)
        with open("{}.json".format(pipeline_id), "w") as f:
            f.write(json.dumps(pipeline, indent=2, default=json_util.default))
            exit(1)
