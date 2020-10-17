import json
import urllib.request

with open('dataset/datafeednetwork2.json') as f:
    data = json.load(f)
print(data)

print(data["Arrow"])

for shape, _ in data.items():
    for _, val in data[shape].items():
        url = val['path']
        urllib.request.urlretrieve(url,
                                   "images/"+shape+
                                   "_"+str(val['minX'])+
                                   "_"+str(val['minY'])+
                                   "_"+str(val['maxX'])+
                                   "_"+str(val['maxY'])+
                                   ".jpg")
        print("downloaded:"+url)