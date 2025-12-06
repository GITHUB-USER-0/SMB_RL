from time import sleep
from PIL import Image
from io import BytesIO
import requests
import pandas as pd

courses = []
courseURLs = []
headers = {'user-agent' : 'dss2q@virginia.edu'}
url = "https://www.vgmaps.com/Atlas/NES/SuperMarioBros-World"

for i in range(1, 9):
    for j in range(1, 5):
        courses.append(f"{i}-{j}")

for course in courses:
    cat = url + course + '.png'
    courseURLs.append(cat)

#for course, url in zip(courses, courseURLs):                            
#        r = requests.get(url, headers = headers)
#        i = Image.open(BytesIO(r.content))
#        i.save(f'{course}.png')
#        sleep(2)

s = ''
s += 'course,width,height\n'

resultdict = {}
for course in courses:
    i = Image.open(f'{course}.png')
    width, height = i.size
    s += (f"{course},{width},{height}\n")
    resultdict[course] = {'width' : width, 'height' : height}

with open('sizes.csv', 'w') as o:
    o.write(s)

foo = pd.read_csv('sizes.csv')
print(foo)
print(resultdict)