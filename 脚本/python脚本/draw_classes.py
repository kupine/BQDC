import json
 
className = {
    1:'person',
    16:'bird',
    17:'cat',
    21:'cow',
    18:'dog',
    19:'horse',
    20:'sheep',
    5:'aeroplane',
    2:'bicycle',
    9:'boat',
    6:'bus',
    3:'car',
    4:'motorbike',
    7:'train',
    44:'bottle',
    62:'chair',
    67:'dining table',
    64:'potted plant',
    63:'sofa',
    72:'tvmonitor'
}
 
classNum = [1,2,3,4,5,6,7,9,16,17,18,19,20,21,44,62,63,64,67,72]
 
def writeNum(Num):
    with open("instances_train2017_1.json","a+") as f:
        f.write(str(Num))
 
# with open("instances_val2014.json","r+") as f:
#     data = json.load(f)
    # annData = data["annotations"]
    # print(annData[0])
    # for x in annData[0]:
    #     if(x == "image_id"):
    #         print(type(x))
    #         print(x+ ":" + str(annData[0][x]))
    #     if (x == "image_id" or x == "bbox" or x == "category_id"):
    #         print(x + ":" + annData[0][x])
    #     if (x == "image_id" or x == "bbox" or x == "category_id"):
    #         print(x+ ":" + annData[0][x])
 
# with open("test.json","w") as f:
#     json.dump(annData, f, ensure_ascii=False)
 
inputfile = []
inner = {}

with open("instances_train2017.json","r+") as f:
    allData = json.load(f)
    data = allData["annotations"]
    print(data[1])
    print("read ready")
 
for i in data:
    if(i['category_id'] in classNum):
        inner = {
            "filename": str(i["image_id"]).zfill(6),
            "name": className[i["category_id"]],
            "bndbox":i["bbox"]
        }
        inputfile.append(inner)
inputfile = json.dumps(inputfile)
writeNum(inputfile)
