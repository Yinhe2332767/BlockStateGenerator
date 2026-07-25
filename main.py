### Script to check if the given two initial configurations will cover all 24 possible rotation states of a block. 
### Outputs T/F check and x-y angles for each configuration to a json file in properformat. 
import json
import numpy as np;

# Input two initial configurations
ini_dirs = [["north", "up"],
            ["north", "east"]]
name = "paired_output"
file_name = name+".json"
model_1_path = "redstoneflightcontrolset:block/"+name
model_2_path = "redstoneflightcontrolset:block/"+name+"_ne"
prop1 = "facing"
prop2 = 'norm'
fillRest = True # If True, fills all impossible states to model_1 to stop Minecraft from reporting error
# Appearantly neoforge has a more elegent way of setting up a default, but now I'm already writing a script to generate instead of hand-write a json...

VECTORS = {
    "east":  [1, 0, 0],
    "south": [0, 0, 1], 
    "up":    [0, 1, 0],
    "west":  [-1, 0, 0],
    "north": [0, 0, -1], 
    "down":  [0, -1, 0]
}
VEC_TO_NAME = {tuple(v): k for k, v in VECTORS.items()}
HELPERVC = {"east":0, "south":1, "up":2, "west":3, "north":4, "down":5}
HELPERVCRV = {0:'east', 1:'south', 2:'up', 3:'west', 4:'north', 5:'down'}
HELPERVCRVO = {0:'west', 1:'north', 2:'down', 3:'east', 4:'south', 5:'up'}

iniDirs = np.array([[VECTORS.get(ini_dirs[0][0]), VECTORS.get(ini_dirs[0][1])], 
                   [VECTORS.get(ini_dirs[1][0]), VECTORS.get(ini_dirs[1][1])]])

x90 = [
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
]
y90 = [
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0]
]
x90 = np.array(x90).transpose()
y90 = np.array(y90).transpose()

def rl(arr): return range(len(arr))

def tryAllStates(mat90, vecs): 
    mat180 = mat90@mat90
    return np.array([
        vecs,
        np.einsum('ij,ykj->yki',mat90,vecs),
        np.einsum('ij,ykj->yki',mat180,vecs),
        np.einsum('ij,ykj->yki',mat90.T,vecs)
    ])

def isRepeated(d1, d2, coveredMsk): # return T/F repeated, also update mask
    idx1 = HELPERVC.get(d1)
    idx2 = HELPERVC.get(d2)
    if coveredMsk[idx1, idx2] == 1: 
        return True
    coveredMsk[idx1, idx2] = 1
    return False

rst = [tryAllStates(x90, iniDirs)]
rst.append(np.einsum('ij,uykj->uyki',y90,rst[0]))
rst.append(np.einsum('ij,uykj->uyki',y90@y90,rst[0]))
rst.append(np.einsum('ij,uykj->uyki',y90.T,rst[0]))

rot_result = [
    [[0,0], [90,0], [180,0], [-90,0]],
    [[0,90], [90,90], [180,90], [-90,90]],
    [[0,180], [90,180], [180,180], [-90,180]], 
    [[0,-90], [90,-90], [180,-90], [-90,-90]]
    ]
rot_result = np.array(rot_result)
outDic = {"variants":{}}
coveredMsk = np.zeros([6,6])
for i in rl(rot_result): 
    for j in rl(rot_result[0]):
        pa = rst[i][j]
        d1 = VEC_TO_NAME.get(tuple(pa[0,0]))
        d2 = VEC_TO_NAME.get(tuple(pa[0,1]))
        if not isRepeated(d1,d2,coveredMsk):
            outDic.get("variants").update({ #type:ignore
                prop1+'='+d1+','+prop2+'='+d2 : #type:ignore
                {'model' : model_1_path, 'x' : int(rot_result[i,j,0]), 'y' : int(rot_result[i,j,1])}
                }) 
        d1 = VEC_TO_NAME.get(tuple(pa[1,0]))
        d2 = VEC_TO_NAME.get(tuple(pa[1,1]))
        if not isRepeated(d1,d2,coveredMsk):
            outDic.get("variants").update({ #type:ignore
                prop1+'='+d1+','+prop2+'='+d2 : #type:ignore
                {'model' : model_2_path, 'x' : int(rot_result[i,j,0]), 'y' : int(rot_result[i,j,1])}
                }) 

cksum = np.sum(coveredMsk)
print('All 24 cases covered:', cksum>=24)
print(cksum, '/24 cases are covered.')
if cksum < 24: 
    print('Cases NOT covered:')
    for i in rl(coveredMsk): 
        for j in rl(coveredMsk[0]): 
            if coveredMsk[i,j] == 0: 
                d1 = HELPERVCRV.get(i)
                d2 = HELPERVCRV.get(j)
                d2o = HELPERVCRVO.get(j)
                if d1 == d2 or d1 == d2o: continue
                print(prop1,': ', d1, '; ', prop2, ': ', d2)

if fillRest: 
    for i in rl(coveredMsk): 
        for j in rl(coveredMsk[0]):
            if coveredMsk[i,j] == 0: 
                d1 = HELPERVCRV.get(i)
                d2 = HELPERVCRV.get(j)
                outDic.get("variants").update({ #type:ignore
                    prop1+'='+d1+','+prop2+'='+d2 : #type:ignore
                    {'model' : model_1_path}})
                coveredMsk[i,j] = 1

with open(file_name, 'w', encoding='utf-8') as f: 
    json.dump(outDic, f, indent=2, ensure_ascii=False)