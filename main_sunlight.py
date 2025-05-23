from utils import *
import numpy as np
import cv2
import warnings
import sys
import random
import time
from matplotlib import pyplot as plt

#濑户内海地图,搜索100次，像素宽500
image = cv2.imread('img_processed.png')
start = (90, 400)
goal = (420, 50)
goal = (200, 250)
enditer=100

# #迷宫地图，搜索400次
# image = cv2.imread('maze_processed.png')
# start = (60, 490)
# goal = (880, 10)
# enditer = 400

# #半圆地图，搜索40次
# image = cv2.imread('circlemap.png')
# start = (320, 490)
# goal = (820, 490)
# enditer = 40

# #半圆地图，搜索100次
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

#判断start和goal是否直接相连，如是，则无需进行规划了
flag , _ =findcross(start, goal, 1, image)
if flag == 0:
    warnings.warn("start和goal之间无障碍物，直接相连", UserWarning)
    cv2.line(image, start, goal, (0, 255, 0), 2)
    cv2.circle(image, start, 5, (255, 0, 0), -1)
    cv2.circle(image, goal, 5, (0, 0, 255), -1)
    cv2.imshow('Planned path', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(1)

#算法开始
#启发函数权重
alpha = 0.8
#贪婪系数
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
    #epsilon概率选取第1个，否则随机选取其他任意1个
    if len(openset)>1:
        if random.uniform(0,1)>epsilon:
            chosennode=random.choice(openset[1:])
        else:
            chosennode=openset[0]
    elif len(openset)==1:
        chosennode=openset[0]
    else:
        print('OPENSET为空，算法失效，请重新调试')
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
    print(f'第{iter}次迭代')

if goalfoundflag==1:
    sonconnectgoal.sort(key=lambda x:x[1])
    print(f'原始规划路径总长{sonconnectgoal[0][1]}')
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
    print(f'原始规划的路径为{plannedpath}')
    endtime = time.time()
    print(f'程序运行时间为{endtime-begintime}s')

    cv2.line(imageb, point, pointf, (0, 0, 255), 3)
    cv2.imshow('Planned path', imageb)
else:
    print('没有搜索到目标点')

#后端处理程序
n=5#后端处理次数，反正反正反。。。
flag=1#下一次的处理处理方式，1为正向处理，-1为反向处理
if goalfoundflag==1:
    plannedpath.reverse()
    L, plannedpath=backendprocess(plannedpath, step, image)
    plannedpath.reverse()
    print(f'第1次后端处理后规划路径总长{L}')
    print(f'第1次后端处理后规划的路径为{plannedpath}')
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

    print(f'第{n}次后端处理后规划路径总长{L}')
    print(f'第{n}次后端处理后规划的路径为{plannedpath}')
    lineplot(plannedpath, imageb)

print(f'程序运行{sampleinstant[-1]}s')
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体
_,ax=plt.subplots()
plt.plot(samplenumber,sampleinstant,'b',linewidth=2)
ax.set_xlabel('Number of sampling')
ax.set_ylabel('Time(s)')
plt.show()

