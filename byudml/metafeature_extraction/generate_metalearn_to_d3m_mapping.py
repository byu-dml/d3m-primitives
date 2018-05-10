import json
import os


def make_json_series(metalearnTitle, d3mTitle, requiredType):
	jsonData = {}
	key = "Mean" + metalearnTitle
	value = {"data_metafeatures_path": d3mTitle + ".mean", "required_type": requiredType}
	jsonData[key] = value
	key = "Stdev" + metalearnTitle
	value = {"data_metafeatures_path": d3mTitle + ".std", "required_type": requiredType}
	jsonData[key] = value
	key = "Min" + metalearnTitle
	value = {"data_metafeatures_path": d3mTitle + ".min", "required_type": requiredType}
	jsonData[key] = value
	key = "Max" + metalearnTitle
	value = {"data_metafeatures_path": d3mTitle + ".max", "required_type": requiredType}
	jsonData[key] = value
	key = "Quartile1" + metalearnTitle
	value = {"data_metafeatures_path": d3mTitle + ".quartile_1", "required_type": requiredType}
	jsonData[key] = value
	key = "Quartile2" + metalearnTitle
	value = {"data_metafeatures_path": d3mTitle + ".quartile_2", "required_type": requiredType}
	jsonData[key] = value
	key = "Quartile3" + metalearnTitle
	value = {"data_metafeatures_path": d3mTitle + ".quartile_3", "required_type": requiredType}
	jsonData[key] = value
	print(json.dumps(jsonData, indent=4))
