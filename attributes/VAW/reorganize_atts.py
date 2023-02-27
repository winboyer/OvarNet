import json

data = json.load(open(r"attributes.json", 'r', encoding='gbk'))
data = [m.split(',')[0] for x in data['attribute_tree'] for m in list(x.values())[0]]
json_data = {'num': len(data), 'attributes': list(data)}
json.dump(json_data, open('../attribute_all/VAW_all_extracted.json', 'w'), indent=4)