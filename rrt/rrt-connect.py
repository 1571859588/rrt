import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
class Node:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.parent=None

class RRT_Connect:
    def sample(self):#返回序列
        if random.randint(0,100)>self.goal_sample_rate:
            #使用序列形式表示随机点，但不是节点，因为从上一个节点到下一个节点的步长为2，而随机生成的点的距离可能大于步长
            rnd=[random.uniform(self.min_rand,self.max_rand),random.uniform(self.min_rand,self.max_rand)]#uniform函数生成实数
        else:
            rnd=[self.goal.x,self.goal.y]
        return rnd
    def get_nearest_node_index(self,rnd,node_list):
        list_len=[(node.x-rnd[0])**2+(node.y-rnd[1])**2 for node in node_list]
        maxIndex=list_len.index(min(list_len))
        return maxIndex

    def get_new_node(self,theta,nearestNode,n_ind):
        newNode=copy.deepcopy(nearestNode)
        newNode.x+=self.expand_dis*math.cos(theta)
        newNode.y+=self.expand_dis*math.sin(theta)
        newNode.parent=n_ind
        return newNode

    def distance_squared_point_to_segment(self,v,w,p):#wvp分别代表最近节点、随机节点、障碍物圆心
        if np.array_equal(v , w):
            return (p-v).dot(p-v)
        l2=(w-v).dot(w-v)#最近节点与随机节点的线段的平方
        t=max(0,min(1,(p-v).dot(w-v)/l2))
        projection=v+t*(w-v)#投影
        return (p-projection).dot(p-projection)#障碍物圆心到线段最短距离

    def check_collision(self,x1,y1,x2,y2):
        for (ox,oy,size) in self.obstacle_list:
            dd=self.distance_squared_point_to_segment(
                np.array([x1,y1]),
                np.array([x2,y2]),
                np.array([ox,oy])
            )
            if dd<=size**2:
                return False #表示两个节点之间存在障碍物

        return True

    def draw_graph(self,newNode=None,path=None,node_list=None):
        plt.clf()
        if newNode is not None:
            plt.plot(newNode.x,newNode.y,"^k") #在新节点处画上黑色的上三角形

        for node in node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x,node_list[node.parent].x],
                             [node.y,node_list[node.parent].y],"-g")#绿色的实线

        for (ox,oy,size) in self.obstacle_list:
            plt.plot(ox,oy,"ok",ms=20*size)#ok表示画黑色的圆形，粗细为20

        #在起点和终点处画红色交叉
        plt.plot(self.start.x,self.start.y,"xr")
        plt.plot(self.goal.x,self.goal.y,"xr")

        if path is not None:
            plt.plot([x for (x,y) in path],[y for (x,y) in path],'-r')#红色实线

        plt.axis([-2,18,-2,18])
        plt.grid(True)
        plt.pause(0.01)
    def line_cost(self,newNode,goal):
        return math.sqrt((newNode.x-goal.x)**2+(newNode.y-goal.y)**2)
    def is_near_goal(self,newNode,end_newNode):
        d=self.line_cost(newNode,end_newNode)
        if d<=self.expand_dis:
            return True
        return False
    def get_final_course(self,lastIndex,node_list,path,start):
        # path=[[self.goal.x,self.goal.y]]
        while node_list[lastIndex].parent is not None:
            node = node_list[lastIndex]
            path.append([node.x,node.y])
            lastIndex=node.parent
        path.append([start.x,start.y])


    def get_path_len(self,path):
        pathlen=0
        for i in range(1,len(path)):
            node1_x=path[i][0]
            node1_y=path[i][1]
            node2_x=path[i-1][0]
            node2_y=path[i-1][1]
            pathlen+=math.sqrt((node1_x-node2_x)**2+(node1_y-node2_y)**2)

        return pathlen
    def __init__(self, randArea, obstacleList,
                 expandDis=2.0, goalSampleRate=10, maxIter=200):
        self.start = None  # 初始化起点
        self.goal = None
        self.max_rand = randArea[0]
        self.min_rand = randArea[1]
        self.expand_dis = expandDis
        self.goal_sample_rate = goalSampleRate
        self.max_iter = maxIter
        self.obstacle_list = obstacleList
        self.start_path = None
        self.end_path=None

        self.start_node_list = None
        self.end_node_list=None

    def planning(self,start,goal,animation=True):
        start_time=time.time()
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.start_node_list = [self.start]
        self.end_node_list=[self.goal]


        for i in range(self.max_iter):
            rnd=self.sample() #随机产生节点序列
            n_ind=self.get_nearest_node_index(rnd,self.start_node_list) #找到距离随机节点序列的最近节点  起始索引
            start_nearestNode=self.start_node_list[n_ind]

            #找出该随机节点序列的步长节点，并判断是否与障碍物发生碰撞

            #通过最近节点与随机节点序列的正切角度，计算出下一节点的位置
            theta=math.atan2((rnd[1]-start_nearestNode.y),(rnd[0]-start_nearestNode.x))

            newNode=self.get_new_node(theta,start_nearestNode,n_ind) #起始节点图的最新节点

            #查看最近节点和新节点连线是否与障碍物碰撞
            noCollision=self.check_collision(newNode.x,newNode.y,start_nearestNode.x,start_nearestNode.y)
            print("测试：")
            if noCollision:
                self.start_node_list.append(newNode)

                if animation:
                    self.draw_graph(newNode,self.start_path,self.start_node_list)



                end_n_ind=self.get_nearest_node_index(newNode,self.end_node_list) #找到距离起始图新节点最近的在终点图的节点

                # print(end_n_ind)
                end_nearestNode=self.end_node_list[end_n_ind]

                theta = math.atan2((newNode[1] - end_nearestNode.y), (newNode[0] - end_nearestNode.x))

                end_newNode = self.get_new_node(theta, end_nearestNode, end_n_ind)  # 起始节点图的最新节点

                # 查看最近节点和新节点连线是否与障碍物碰撞
                end_noCollision = self.check_collision(end_newNode.x, end_newNode.y, end_nearestNode.x, end_nearestNode.y)

                if end_noCollision:
                    self.end_node_list.append(end_newNode)
                if animation:
                    self.draw_graph(newNode,self.end_path,self.end_node_list)
                while True:
                    end_n_ind = self.get_nearest_node_index(newNode, self.end_node_list)  # 找到距离起始图新节点最近的在终点图的节点
                    theta = math.atan2((end_newNode[1] - newNode[1]), (end_newNode[0] - newNode[0]))
                    end_nearestNode = self.end_node_list[end_n_ind]
                    NEWnewNode = self.get_new_node(theta, end_nearestNode, end_n_ind)  # 起始节点图的最新节点

                    # 查看最近节点和新节点连线是否与障碍物碰撞
                    NOCollision = self.check_collision(NEWnewNode.x, NEWnewNode.y, end_nearestNode.x, end_nearestNode.y)
                    if NOCollision:
                        self.end_node_list.append(NEWnewNode)
                        if animation:
                            self.draw_graph(NEWnewNode, self.end_path, self.end_node_list)
                        end_newNode=copy.deepcopy(NEWnewNode)
                    else:
                        break
                    if end_newNode == newNode:
                        break
                if end_newNode == newNode or self.is_near_goal(end_newNode,newNode):

                    if self.check_collision(newNode.x,newNode.y,end_newNode.x,end_newNode.y):

                        start_lastIndex=len(self.start_node_list)-1
                        end_lastIndex=len(self.end_node_list)-1
                        self.get_final_course(start_lastIndex,self.start_node_list,self.start_path,self.start)
                        self.get_final_course(end_lastIndex, self.end_node_list, self.end_path, self.goal)

                        pathlen=self.get_path_len(self.start_path)+self.get_path_len(self.end_path)
                        print("current path length:{},It costs {} s".format(pathlen,time.time()-start_time))

                        if animation:
                            self.draw_graph(newNode,self.start_path,self.start_node_list)
                            self.draw_graph(newNode, self.end_path,self.end_node_list)
                            return self.start_path+self.end_path
            if len(self.start_node_list)>len(self.end_node_list):
                node_list=copy.deepcopy(self.start_node_list)
                self.start_node_list=copy.deepcopy(self.end_node_list)
                self.end_node_list=copy.deepcopy(node_list)


def mn():
    show=True
    print("start rrt-connect planning")
    obstacleList=[(3,3,1.5),(12,2,5),(3,9,2),(9,11,2)]
    rrt=RRT_Connect(randArea=[-2,18],obstacleList=obstacleList,maxIter=300)
    path=rrt.planning(start=[0,0],goal=[11,13],animation=show)
    print("Done!")
    if path and show:
        plt.show()
