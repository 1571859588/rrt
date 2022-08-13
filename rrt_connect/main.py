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
        # plt.clf()
        if newNode is not None:
            plt.plot(newNode.x,newNode.y,"^k") #在新节点处画上黑色的上三角形

        for node in node_list:
            if node is not None:
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
        plt.pause(0.1)
    def line_cost(self,newNode,goal):
        return math.sqrt((newNode.x-goal.x)**2+(newNode.y-goal.y)**2)
    def is_near(self,newNode,end_newNode):
        d=self.line_cost(newNode,end_newNode)
        if d<=self.expand_dis:
            return True
        return False
    def get_final_course(self,lastIndex,node_list,path,start,goal):
        path=[[goal.x,goal.y]]

        while node_list[lastIndex].parent is not None:
            node = node_list[lastIndex]
            path.append([node.x,node.y])
            lastIndex=node.parent
        path.append([start.x,start.y])
        # path.pop(0)
        return path

    def get_path_len(self,num):
        pathlen=0
        if num==0:
            path=self.start_path
        else:
            path=self.end_path
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
        #合并的路径和节点
        self.path=None
        self.node_list=None

        self.start_node_list = None
        self.end_node_list=None
    def get_newNode(self,rnd,node_list):

        n_ind = self.get_nearest_node_index(rnd, node_list)  # 找到距离随机节点序列的最近节点  起始索引
        nearestNode = node_list[n_ind]

        # 找出该随机节点序列的步长节点，并判断是否与障碍物发生碰撞

        # 通过最近节点与随机节点序列的正切角度，计算出下一节点的位置
        theta = math.atan2((rnd[1] - nearestNode.y), (rnd[0] - nearestNode.x))

        newNode = self.get_new_node(theta, nearestNode, n_ind)  # 起始节点图的最新节点
        return newNode
    def get_collision(self,rnd,node_list):
        n_ind = self.get_nearest_node_index(rnd, node_list)  # 找到距离随机节点序列的最近节点  起始索引
        nearestNode = node_list[n_ind]

        # 找出该随机节点序列的步长节点，并判断是否与障碍物发生碰撞

        # 通过最近节点与随机节点序列的正切角度，计算出下一节点的位置
        theta = math.atan2((rnd[1] - nearestNode.y), (rnd[0] - nearestNode.x))

        newNode = self.get_new_node(theta, nearestNode, n_ind)  # 起始节点图的最新节点
        return self.check_collision(newNode.x, newNode.y, nearestNode.x, nearestNode.y)
    def planning(self,start,goal,animation=True):
        start_time=time.time()
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.GOAL=goal
        self.start_node_list = [self.start]
        self.end_node_list=[self.goal]


        for i in range(self.max_iter):
            rnd=self.sample() #随机产生节点序列
            newNode=self.get_newNode(rnd,self.start_node_list)

            #查看最近节点和新节点连线是否与障碍物碰撞
            noCollision=self.get_collision(rnd,self.start_node_list)
            # print("测试：",i)
            if noCollision:
                self.start_node_list.append(newNode)

                if animation:
                    self.draw_graph(newNode,self.start_path,self.start_node_list)
                rnd=[newNode.x,newNode.y]
                # print(rnd)
                newNode1 = self.get_newNode(rnd, self.end_node_list) #终点图的新节点

                # 查看最近节点和新节点连线是否与障碍物碰撞
                noCollision1 = self.get_collision(rnd, self.end_node_list)

                if noCollision1:
                    self.end_node_list.append(newNode1)
                if animation:
                    self.draw_graph(newNode,self.end_path,self.end_node_list)

                while True:

                    rnd = [newNode.x, newNode.y]
                    newNode2 = self.get_newNode(rnd, self.end_node_list)  # 终点图的新节点

                    # 查看最近节点和新节点连线是否与障碍物碰撞
                    noCollision2 = self.get_collision(rnd, self.end_node_list)
                    if noCollision2:
                        self.end_node_list.append(newNode2)
                        if animation:
                            self.draw_graph(newNode2, self.end_path, self.end_node_list)
                        # end_newNode=copy.deepcopy(newNode2)
                        newNode1=newNode2
                    else:
                        break
                    if self.is_near(newNode1,newNode):
                        if self.check_collision(newNode.x, newNode.y, newNode1.x, newNode1.y):

                            start_lastIndex = len(self.start_node_list) - 1
                            end_lastIndex = len(self.end_node_list) - 1

                            self.start_path=self.get_final_course(start_lastIndex, self.start_node_list, self.start_path, self.start,self.goal)
                            # print(self.start_path)
                            self.start_path.pop(0)
                            # print(self.start_path)
                            self.end_path=self.get_final_course(end_lastIndex, self.end_node_list, self.end_path, self.goal,self.start)
                            self.end_path.pop(0)


                            pathlen = self.get_path_len(0) + self.get_path_len(1)
                            print("current path length:{},It costs {} s".format(pathlen, time.time() - start_time))

                            if animation:
                                # self.draw_graph(None, self.start_path, self.start_node_list)
                                # self.draw_graph(None, self.end_path, self.end_node_list)
                                print("start_path:",self.start_path)
                                print("end_path:",self.end_path)

                                # for i in range(len(self.start_node_list)):
                                #     print("start_node_list:",self.start_node_list[i].x,self.start_node_list[i].y)
                                #
                                # for i in range(len(self.end_node_list)):
                                #     print("end_node_list:",self.end_node_list[i].x,self.end_node_list[i].y)
                                # # #合并
                                # self.node_list=[Node(0,0)]
                                # #考虑终点图最后一个点与起始图最后一个点是否可以连接
                                while True:

                                    if self.start_path[0] == [0, 0]:

                                        if self.end_path[len(self.end_path) - 1] == self.GOAL:

                                            break
                                        else:
                                            self.end_path.reverse()
                                    else:
                                        if self.start_path[0] != [0, 0] and self.start_path[
                                            len(self.start_path) - 1] != [0, 0]:
                                            path = self.end_path
                                            self.end_path = self.start_path
                                            self.start_path = path
                                        else:
                                            self.start_path.reverse()
                                length_start=len(self.start_path)
                                if self.is_near(Node(self.start_path[length_start-2][0],self.start_path[length_start-2][1]),Node(self.end_path[0][0],self.end_path[0][1])) and self.is_near(Node(self.start_path[length_start-2][0],self.start_path[length_start-2][1]),Node(self.end_path[1][0],self.end_path[1][1])):
                                    self.end_path.pop(0)
                                if self.is_near(Node(self.start_path[length_start-2][0],self.start_path[length_start-2][1]),Node(self.end_path[0][0],self.end_path[0][1])) and self.is_near(Node(self.start_path[length_start-2][0],self.start_path[length_start-2][1]),Node(self.end_path[0][0],self.end_path[0][1])):
                                    self.start_path.pop()

                                #
                                # self.end_node_list.reverse()
                                # self.node_list=self.start_node_list+self.end_node_list
                                # #
                                # for i in range(len(self.node_list)):
                                #     print("node_list:",self.node_list[i].x,self.node_list[i].y)
                                #
                                self.path=[[0,0]]
                                # self.start_path.sort()
                                # self.end_path.sort()
                                # if (self.start_path[0] or self.start_path[len(self.start_path)-1]==[0,0]):
                                self.path = self.start_path + self.end_path


                                # print("node_list:",self.node_list)


                                print("path:",self.path)
                                #
                                #
                                #
                                #
                                #
                                self.draw_graph(None,self.path,self.start_node_list)
                                return self.start_path + self.end_path
                        break
                # if (newNode1.x==newNode.x and newNode1.y==newNode.y ) or self.is_near(newNode1,newNode):
                #
                #     if self.check_collision(newNode.x,newNode.y,newNode1.x,newNode1.y):
                #
                #         start_lastIndex=len(self.start_node_list)-1
                #         end_lastIndex=len(self.end_node_list)-1
                #         self.get_final_course(start_lastIndex,self.start_node_list,self.start_path,self.start)
                #         self.get_final_course(end_lastIndex, self.end_node_list, self.end_path, self.goal)
                #
                #         pathlen=self.get_path_len(self.start_path)+self.get_path_len(self.end_path)
                #         print("current path length:{},It costs {} s".format(pathlen,time.time()-start_time))
                #
                #         if animation:
                #             self.draw_graph(newNode,self.start_path,self.start_node_list)
                #             self.draw_graph(newNode, self.end_path,self.end_node_list)
                #             return self.start_path+self.end_path
            if len(self.start_node_list)>len(self.end_node_list):
                # print("转换")
                # print(len(self.start_node_list))
                # print(len(self.end_node_list))
                #
                # node_list=copy.deepcopy(self.start_node_list)
                # self.start_node_list=copy.deepcopy(self.end_node_list)
                # self.end_node_list=copy.deepcopy(node_list)


                path=self.start_path
                self.start_path=self.end_path
                self.end_path=path

                node_list=self.start_node_list
                self.start_node_list=self.end_node_list
                self.end_node_list=node_list
                # print(len(self.start_node_list))
                # print(len(self.end_node_list))
                GOAL=self.goal
                self.goal=self.start
                self.start = GOAL

def main():
    show=True
    print("start rrt-connect planning")
    obstacleList=[(3,3,1.5),(12,2,5),(3,9,2),(9,11,2)]
    rrt=RRT_Connect(randArea=[-2,18],obstacleList=obstacleList,maxIter=300)
    path=rrt.planning(start=[0,0],goal=[11,13],animation=show)
    print("Done!")
    if path and show:
        plt.show()
if __name__ == '__main__':
    main()
