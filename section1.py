import json
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from pycocotools.coco import COCO
coco = COCO("./annotations/person_keypoints_val2017.json")

cats = coco.loadCats(coco.getCatIds())
catIds = coco.getCatIds(catNms=["person"])
imgIds = coco.getImgIds(catIds=catIds)

img = coco.loadImgs(imgIds[0])
ann = coco.loadAnns(coco.getAnnIds(imgIds=imgIds[0])[0])
I = Image.open("./val2017/" + img[0]["file_name"])
I_crop = I.crop((ann[0]["bbox"][0],ann[0]["bbox"][1],ann[0]["bbox"][0] + ann[0]["bbox"][2],ann[0]["bbox"][1] + ann[0]["bbox"][3]))
out = I_crop.resize((256,192))
d = ImageDraw.Draw(out)
j = 0
keypointsArray = []
for i in cats[0]["keypoints"]:
    if ann[0]["keypoints"][2+j*3] > 0:
        x = (ann[0]["keypoints"][j*3] - ann[0]["bbox"][0])/ann[0]["bbox"][2] * 256
        y = (ann[0]["keypoints"][1 + j*3] - ann[0]["bbox"][1])/ann[0]["bbox"][3] * 192
        d.text((x,y),i,fill=(255,0,0,255))
        d.point((x,y),fill=(255,0,0,255))
    j+=1
out.show()
