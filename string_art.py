import numpy as np
import cv2 as cv
import math
import pybresenham
import copy
import matplotlib.pylab as plt

#x direction = height, y direction = wide
#pointnum:number of point, radius:radius of circle, point:[x,y] of the points, numline: number of lines
pointnum = 256
size_w, size_h = 750, 750
radius = 300
numline = 1000
input_image = cv.imread("testimage4.jpg")
#input_image = cv.imread("testimage4.jpg")


#color: dict
color = {"black":(0,0,0), "white":(255,255,255)}

#drawing line
def line(img, p1, p2, c, d):
    img = cv.line(img, p1,p2,c,d)
    return img

#bresenham algorithm
def find_point_on_line(p1, p2):
    line_point = list(pybresenham.line(p1[0],p1[1],p2[0],p2[1]))
    return line_point


def greedy_line(img_gray, current_point_index, point):
    #not to connect close points
    minium_index = 10
    minsum = 255
    for i in range(pointnum):
        dis = abs(i - current_point_index)
        if (dis <= minium_index) or (256-dis <= minium_index):
            continue
        
        line_point = find_point_on_line(point[current_point_index] , point[i])
        
        sum_gray = 0
        for j in range(len(line_point)):
            sum_gray += img_gray[line_point[j][0],line_point[j][1]]
        average = sum_gray / len(line_point)
        
        if average < minsum:
            minsum = average
            next_point_index = i

    
    return next_point_index

def greedy_line_white(img_gray, current_point_index, point):
    #not to connect close points
    minium_index = 10
    minsum = 0
    for i in range(pointnum):
        dis = abs(i - current_point_index)
        if (dis <= minium_index) or (256-dis <= minium_index):
            continue
        
        line_point = find_point_on_line(point[current_point_index] , point[i])
        
        sum_gray = 0
        for j in range(len(line_point)):
            sum_gray += img_gray[line_point[j][0],line_point[j][1]]
        average = sum_gray / len(line_point)
        
        if average > minsum:
            minsum = average
            next_point_index = i

    return next_point_index

# def greedy_point(img_gray, current_point_index, point):
#     mini

def weaver_line(img, img_gray, point):
    current_point_index = 0
    d = 1

    for num in range(numline):
        next_point_index = greedy_line(img_gray, current_point_index, point)
        img = cv.line(img, [point[current_point_index][1],point[current_point_index][0]], [point[next_point_index][1],point[next_point_index][0]], color['black'], d)
        
        line_point = find_point_on_line(point[current_point_index] , point[next_point_index])
        for j in range(len(line_point)):
            img_gray[line_point[j][0],line_point[j][1]] = 255

        current_point_index = next_point_index

        cv.imshow('image',img)
        cv.imshow('rrr',img_gray)
        cv.waitKey(1)
        
    return img

def weaver_line_start_npoint(img, img_gray, point):
    current_point_index = [0,128]
    next_point_index = [0,0]
    d = 1
    n = 2 

    for num in range(numline//n):
        for i in range(n):
            next_point_index[i] = greedy_line(img_gray, current_point_index[i], point)
            img = cv.line(img, [point[current_point_index[i]][1],point[current_point_index[i]][0]], [point[next_point_index[i]][1],point[next_point_index[i]][0]], color['black'], d)
            
            line_point = find_point_on_line(point[current_point_index[i]] , point[next_point_index[i]])
            for j in range(len(line_point)):
                img_gray[line_point[j][0],line_point[j][1]] = 255

            current_point_index[i] = next_point_index[i]

            cv.imshow('image',img)
            cv.imshow('rrr',img_gray)
            cv.waitKey(1)
        
    return img



def imagecutting(inimage):
    x1, y1 = len(inimage)//2 - size_h//2, len(inimage[0])//2 - size_w//2
    x2, y2 = len(inimage)//2 + size_h//2, len(inimage[0])//2 + size_w//2
    
    img_cut = np.zeros(shape=(size_h,size_w,3), dtype=np.uint8)+255
    img_cut = cv.circle(img_cut,(size_h//2,size_w//2),radius,color['black'],-1)

    inimage = inimage[110:860, y1:y2, :]

    #get image in circle
    data1 = cv.bitwise_or(inimage,img_cut)
    data2 = cv.cvtColor(data1, cv.COLOR_BGR2GRAY)
    return data1, data2

def compare_image(img_original, img_compare):
    error_value = 0
    for i in range(len(img_original)):
        for j in range(len(img_original[0])):
            error_value += (img_original[i][j]-img_compare[i][j])**2

    return error_value


def background():
    img = np.zeros(shape=(size_h,size_w,3), dtype=np.uint8) + 255 #white background
        
    #n polynomial pointing
    point = []

    for i in range(0, pointnum):
        deg = float(math.pi * 2 * i / pointnum)
        point_x = int(size_h//2 - radius*math.cos(deg))
        point_y = int(size_w//2 + radius*math.sin(deg))

        point.append([point_x, point_y])
    
    points = np.array(point, np.int32)
    #img = cv.polylines(img, [points], True, color['black'])
    img_color, img_gray = imagecutting(input_image)
    img_gray_2 = copy.deepcopy(img_gray)
    img_gray_3 = copy.deepcopy(img_gray)
    cv.imshow('image',img)
    cv.imshow('rrr',img_gray)
    cv.waitKey(3000)
    cv.destroyAllWindows()
    
    img_2 = copy.deepcopy(img)
    img = weaver_line(img, img_gray, point)
    img_n = weaver_line_start_npoint(img_2, img_gray_2, point)
    error1 = compare_image(img_gray_3,img)
    error2 = compare_image(img_gray_3, img_n)
    print(error1[1], error2[1])
    cv.waitKey(0)
    #weaver_line_white(img, img_gray_2, point)
    #weaver_point(img, img_gray, point)

        
background()