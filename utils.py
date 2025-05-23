import numpy as np
import cv2
import warnings
import sys

def findclosest(point,theta,step,image):
    # 给予point，角度，步长，寻找point在该角度下在image上最近的轮廓点坐标
    if image[point[1],point[0]][1] == 0:
        warnings.warn("point在障碍物内部，请输入正确的point位置！", UserWarning)
        cv2.circle(image, point, 2, (0, 0, 255), -1)
        cv2.imshow('image', image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        sys.exit(1)
    else:
        spoint = point
        i = 0
        while image[spoint[1],spoint[0]][1] != 0 and spoint[0]<image.shape[1] and spoint[0]>=0 and spoint[1]<image.shape[0] and spoint[1]>=0:
            i += 1
            spoint=[int(point[0]+i*step*np.cos(theta)),int(point[1]+i*step*np.sin(theta))]
        spoint=[int(point[0]+(i-1)*step*np.cos(theta)),int(point[1]+(i-1)*step*np.sin(theta))]
        L=np.sqrt((point[0]-spoint[0])**2+(point[1]-spoint[1])**2)
        return spoint, L

def findcircle(point,jstep,step,image):
    # 给予point，角度步长，光线步长，寻找以point为圆心散射在image上的所有轮廓点
    spoints=[]
    for i in np.arange(0,2*np.pi+jstep,jstep):
        spoint,L=findclosest(point, i, step, image)
        temp=[spoint,L,i,point]
        spoints.append(temp)
    return spoints

def findtangent(point,jstep,step,image):
    # 给予point，角度步长，光线步长，寻找以point为圆心在轮廓上的切点，近端（远端省略）
    spoints = findcircle(point,jstep,step,image)
    #切点前向距离
    Delta=15*step
    # Delta = 11 * step
    #切点判断阈值，认为两条相邻射线相差X时，存在切点
    X=15
    L1 = spoints[0][1]
    tagentpoints=[]
    for i in range(1,len(spoints)):
        deltaL=np.abs(spoints[i][1]-L1)
        L1=spoints[i][1]
        if deltaL>=X:
            if spoints[i][1]>spoints[i-1][1]:
                spointless=[[int(point[0]+(spoints[i-1][1]+Delta)*np.cos(spoints[i][2])),int(point[1]+(spoints[i-1][1]+Delta)*np.sin(spoints[i][2]))],spoints[i-1][1]+Delta,spoints[i][2],point]
            else:
                spointless =[[int(point[0]+(spoints[i][1]+Delta)*np.cos(spoints[i-1][2])),int(point[1]+(spoints[i][1]+Delta)*np.sin(spoints[i-1][2]))],spoints[i][1]+Delta,spoints[i-1][2],point]
            if spointless[0][0]<image.shape[1]-1 and spointless[0][0]>=0 and spointless[0][1]<image.shape[0]-1 and spointless[0][1]>=0 and image[spointless[0][1], spointless[0][0]][1] != 0:
                # print(image[spointless[0][1], spointless[0][0]])
                tagentpoints.append(spointless)
    return tagentpoints

def findcross(point1,point2,step,image):
    # 给予point1，point2，光线步长，寻找连线在轮廓上的最近交点
    spoint=[None,None]
    crossflag=0
    L = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    for i in np.arange(0,L,step):
        spoint[0]=int(point1[0]+i*(point2[0]-point1[0])/L)
        spoint[1]=int(point1[1]+i*(point2[1]-point1[1])/L)
        if image[spoint[1]][spoint[0]][1]==0:
            crossflag=1
            break
    return crossflag, spoint

def costfunction(point,father,goal,fs_accumlength,alpha):
    #给予point,father,goal,father与start的累积长度，alpha，计算cost
    pg_distance = np.sqrt((point[0] - goal[0]) ** 2 + (point[1] - goal[1]) ** 2)
    if father is not None:
        pf_distance=np.sqrt((point[0]-father[0])**2 + (point[1]-father[1])**2)
        cost=fs_accumlength + pf_distance + pg_distance * alpha
    else:
        cost=fs_accumlength + pg_distance * alpha
    return cost

def distantdistance(point1,point2):
    #用于计算两个直接相连point的直线距离
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def newsonfind(father,son,grandson,step,image):
    #后端处理子程序，进一步缩短路径，输入父亲、儿子、孙子的坐标系，调整儿子的位置
    crossflag,_=findcross(father,grandson,step,image)
    if crossflag==0:
        son_new=(int((father[0]+grandson[0])/2),int((father[1]+grandson[1])/2))
    else:
        a = father
        b = son
        while distantdistance(a,b)>2:
            med = (int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2))
            crossflag, _ = findcross(grandson, med, step, image)
            if crossflag ==1:
                a = med
            else:
                b = med
        son_new = b
    return son_new

def backendprocess(plannedpath,step,image):
    # 后端处理程序,输入plannedpath，调用newsonfind，得到新的path
    L = 0
    path = plannedpath
    for i in range(1, len(path) - 1):
        son_new = newsonfind(path[i - 1], path[i], path[i + 1], step, image)
        path[i] = son_new
        L = L + distantdistance(path[i - 1], path[i])
    L = L + distantdistance(path[i], path[i + 1])
    return L, path

def lineplot(plannedpath,image,color=(0,255,0)):
    # 在image上绘制plannedpath的连线
    for i in range(len(plannedpath)-1):
        cv2.line(image, plannedpath[i], plannedpath[i+1], color, 2)
    cv2.imshow('Planned path',image)
    cv2.waitKey(0)
