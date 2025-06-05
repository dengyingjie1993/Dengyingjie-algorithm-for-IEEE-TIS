from utils_english_version import *
import numpy as np
import cv2
import warnings
import sys
import random
import time
from matplotlib import pyplot as plt

#Ito inland sea, search 100 times，image width 500
image = cv2.imread('img_processed.png')
start = (90, 400)
goal = (420, 50)
goal = (200, 250)
enditer=100

# #maze，search 400 times
# image = cv2.imread('maze_processed.png')
# start = (60, 490)
# goal = (880, 10)
# enditer = 400

# #semicircle，search 40 times
# image = cv2.imread('circlemap.png')
# start = (320, 490)
# goal = (820, 490)
# enditer = 40

# #multiple geometries，search 100 times
# image = cv2.imread('circlemap_processed.png')
# start = (30, 490)
# goal = (870, 490)
# enditer = 100

begintime=time.time()
sampleinstant=[]
samplenumber=[]

imageb=image.copy()
cv2.circle(imageb, start, 5, (0, 255, 0), -1)
cv2.circle(imageb, goal, 5, (0, 0, 255), -1)

#judge whether start and goal are connected, if so, path planning is not required
flag , _ =findcross(start, goal, 1, image)
if flag == 0:
    warnings.warn("start and goal are connected without obstacles", UserWarning)
    cv2.line(image, start, goal, (0, 255, 0), 2)
    cv2.circle(image, start, 5, (255, 0, 0), -1)
    cv2.circle(image, goal, 5, (0, 0, 255), -1)
    cv2.imshow('Planned path', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(1)

#Algorithm begin
#weight of heuristic function
alpha = 0.8
#greedy coefficient
epsilon=0.8

step=1
jstep=np.pi/360
iter=0


startcost=costfunction(start,None,goal,0, alpha)
startpoint={'坐标':start,'父亲':None,'accumlength':0,'cost':startcost}
openset=[startpoint]
rootset=[]
goalfoundflag=0
sonconnectgoal = []

while iter<enditer:
    #choose the first one with the probability of epsilon, otherwise choose the random one
    if len(openset)>1:
        if random.uniform(0,1)>epsilon:
            chosennode=random.choice(openset[1:])
        else:
            chosennode=openset[0]
    elif len(openset)==1:
        chosennode=openset[0]
    else:
        print('OPENSET is null, path planning is unavailable')
        sys.exit(1)

    openset.remove(chosennode)
    rootset.append(chosennode)
    cv2.circle(imageb, chosennode['坐标'], 3, (0, 0, 255), -1)

    sons=findtangent(chosennode['坐标'],jstep,step,image)
    sonscoordinate = []
    for son in sons:
        sonaccumlength = son[1]+chosennode['accumlength']
        sonofothersflag = 0
        if chosennode['父亲'] is not None:
            nonbrotherflag , _ =findcross(chosennode['父亲'],son[0],step,image)
        else:
            nonbrotherflag = 1

        for node in openset:
            nonopenpointconnectflag , _ = findcross(node['坐标'], son[0], step, image)
            if nonopenpointconnectflag==0:
                L=distantdistance(node['坐标'],son[0])+node['accumlength']
                if L < sonaccumlength:
                    sonofothersflag = 1
                    break

        if sonofothersflag==0 and nonbrotherflag == 1:
            soncost=costfunction(son[0],chosennode['坐标'],goal,chosennode['accumlength'], alpha)
            newopenpoint={'坐标':son[0],'父亲':chosennode['坐标'],'accumlength':sonaccumlength,'cost':soncost}
            openset.append(newopenpoint)

            goalnotconnectflag , _=findcross(son[0],goal,step,image)
            if goalnotconnectflag == 0:
                goalfoundflag=1
                D=distantdistance(son[0],goal)+sonaccumlength
                sonconnectgoal.append([newopenpoint,D])

            sonscoordinate.append(son[0])
            # cv2.line(imageb, chosennode['坐标'] , son[0], (0, 255, 0), 2)
            cv2.circle(imageb, son[0], 3, (255, 255, 0), -1)

    cv2.imshow('Searching process', imageb)
    cv2.waitKey(10)
    for son in sonscoordinate:
        cv2.circle(imageb, son, 3, (255, 0, 0), -1)
    cv2.circle(imageb, chosennode['坐标'], 3, (0, 76, 153), -1)

    openset.sort(key=lambda x: x['cost'])
    iter=iter+1

    samplenumber.append(len(rootset) + len(openset) - 1)
    sampleinstant.append(time.time() - begintime)
    print(f'the {iter}th iteration')

if goalfoundflag==1:
    sonconnectgoal.sort(key=lambda x:x[1])
    print(f'the length of the initally planned path {sonconnectgoal[0][1]}')
    point=sonconnectgoal[0][0]['坐标']
    pointf=sonconnectgoal[0][0]['父亲']
    plannedpath = [pointf,point,goal]
    cv2.destroyAllWindows()
    cv2.line(imageb, goal, point, (0, 0, 255), 2)
    cv2.imshow('Planned path', imageb)
    cv2.waitKey(100)
    while pointf!=start:
        cv2.line(imageb, point, pointf, (0, 0, 255), 2)
        cv2.imshow('Planned path', imageb)
        cv2.waitKey(100)
        fatherindex = [index for index, value_in_list in enumerate(rootset) if value_in_list['坐标'] == pointf]
        point=pointf
        pointf = rootset[fatherindex[0]]['父亲']
        plannedpath.insert(0,pointf)
    print(f'the initially planned path is {plannedpath}')
    endtime = time.time()
    print(f'the running time is {endtime-begintime}s')

    cv2.line(imageb, point, pointf, (0, 0, 255), 3)
    cv2.imshow('Planned path', imageb)
else:
    print('the goal is not searched')

#Backend subprogram
n=5#the times of backend process, reverse forward reverse forward reverse...
flag=1#the manner for the next process, 1 denotes the forward, -1 denotes the reverse
if goalfoundflag==1:
    plannedpath.reverse()
    L, plannedpath=backendprocess(plannedpath, step, image)
    plannedpath.reverse()
    print(f'the path length after the 1st backend process is {L}')
    print(f'the path after the 1st backend process is {plannedpath}')
    # lineplot(plannedpath, imageb,color=(255,0,0))

    for i in range(2,n+1):
        if flag==1:
            L, plannedpath = backendprocess(plannedpath, step, image)
            flag=-1
        else:
            plannedpath.reverse()
            L, plannedpath = backendprocess(plannedpath, step, image)
            plannedpath.reverse()
            flag=1

    print(f'the path length after the {n}th backend process is {L}')
    print(f'the path after the {n}th backend process is {plannedpath}')
    lineplot(plannedpath, imageb)

print(f'the algorithm runs for {sampleinstant[-1]}s')
plt.rcParams['font.family'] = 'Times New Roman'  # set the Font
_,ax=plt.subplots()
plt.plot(samplenumber,sampleinstant,'b',linewidth=2)
ax.set_xlabel('Number of sampling')
ax.set_ylabel('Time(s)')
plt.show()