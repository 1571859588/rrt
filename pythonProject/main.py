import copy
import math
import random
import time

import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as Rot # 三维旋转计算模块
import numpy as np # 向量计算模块

show_animation = True


class RRT:

    def __init__(self, obstacleList, randArea,
                 expandDis=2.0, goalSampleRate=10, maxIter=200):

        self.start = None   # 初始化起点
        self.goal = None    # 初始化终点
        self.min_rand = randArea[0] # 最小边界
        self.max_rand = randArea[1] # 最大边界
        self.expand_dis = expandDis # 初始化步长，默认为2
        self.goal_sample_rate = goalSampleRate  # 初始化以终点为随机点的概率，默认为%10
        self.max_iter = maxIter # 最大迭代次数，默认为200
        self.obstacle_list = obstacleList   # 障碍物列表
        self.node_list = None   # 初始化rrt树节点

    def rrt_planning(self, start, goal, animation=True):
        start_time = time.time()    # 记录开始时间
        self.start = Node(start[0], start[1])   # 设置开始节点
        self.goal = Node(goal[0], goal[1])  # 设置终点
        self.node_list = [self.start]   # 添加rrt树的第一个节点
        path = None # 路径为空

        for i in range(self.max_iter):  # 开始迭代
            rnd = self.sample() # 随机采样生成节点
            n_ind = self.get_nearest_list_index(self.node_list, rnd)    # rrt树上距离随机采样点最近的节点的索引，node_list为rrt树节点集合
            nearestNode = self.node_list[n_ind] # 标记最近节点

            # steer
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)  # 获取最近节点到采样点的连线与水平坐标轴x轴之间的tan值
            newNode = self.get_new_node(theta, n_ind, nearestNode)  # 计算新节点位置

            noCollision = self.check_segment_collision(newNode.x, newNode.y, nearestNode.x, nearestNode.y) # 检测是否新节点与最近节点的连线碰撞
            if noCollision:
                self.node_list.append(newNode) # 无碰撞，将新节点加入rrt树集合中
                if animation:
                    self.draw_graph(newNode, path) # 分步骤画出rrt树的迭代

                if self.is_near_goal(newNode):  # 检测新节点是否在终点的步长范围
                    if self.check_segment_collision(newNode.x, newNode.y,
                                                    self.goal.x, self.goal.y): # 检测新节点终点的连线上是否存在障碍物
                        lastIndex = len(self.node_list) - 1 # 获取最新节点的索引
                        path = self.get_final_course(lastIndex) # 获取解的路径
                        pathLen = self.get_path_len(path) # 获取路径的长度
                        print("current path length: {}, It costs {} s".format(pathLen, time.time()-start_time))

                        if animation:
                            self.draw_graph(newNode, path)
                        return path


    def sample(self):
        if random.randint(0, 100) > self.goal_sample_rate:  # 用randint函数生成0-100内的数，小于等于10则以终点为随机采样点
            rnd = [random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand)]  # uniform函数可以在指定范围内生成一个实数
        else:  # goal point sampling
            rnd = [self.goal.x, self.goal.y]
        return rnd





    @staticmethod
    def get_path_len(path): # 路径长度计算
        pathLen = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            pathLen += math.sqrt((node1_x - node2_x)
                                 ** 2 + (node1_y - node2_y) ** 2)

        return pathLen

    @staticmethod
    def line_cost(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    @staticmethod
    def get_nearest_list_index(nodes, rnd):
        dList = [(node.x - rnd[0]) ** 2
                 + (node.y - rnd[1]) ** 2 for node in nodes]    # 遍历树上的节点到采样点的距离，生成列表
        minIndex = dList.index(min(dList))  # 记录最小距离节点索引
        return minIndex

    def get_new_node(self, theta, n_ind, nearestNode):
        newNode = copy.deepcopy(nearestNode) # 深拷贝最近节点

        newNode.x += self.expand_dis * math.cos(theta)
        newNode.y += self.expand_dis * math.sin(theta)  # 用tan值计算新节点位置

        newNode.parent = n_ind  # 记录新节点的父节点的索引
        return newNode

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)#节点与终点的距离
        if d <= self.expand_dis:
            return True
        return False


    @staticmethod
    def distance_squared_point_to_segment(v, w, p):
        # Return minimum distance between line segment vw and point p
        if np.array_equal(v, w): # 如果新节点v和最近节点重合，直接使用向量内积的方式返回障碍物圆心到点的距离
            return (p - v).dot(p - v)  # dot函数返回向量内积
            # v == w case
        l2 = (w - v).dot(w - v)     # 计算两节点线段距离
        # i.e. |w-v|^2 -  avoid a sqrt
        # Consider the line extending the segment,
        # parameterized as v + t (w - v).
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        # We clamp t from [0,1] to handle points outside the segment vw.
        t = max(0, min(1, (p - v).dot(w - v) / l2)) # 采用向量内积的方式，计算得出（新节点到圆心的向量）在（新节点到最近节点的向量）的投影与（新节点到最近节点的向量）的模的比值
        projection = v + t * (w - v)  # Projection falls on the segment # 计算圆心到线段上最短距离的点
        return (p - projection).dot(p - projection) # 返回圆心到线段最短距离

    def check_segment_collision(self, x1, y1, x2, y2):
        for (ox, oy, size) in self.obstacle_list:
            dd = self.distance_squared_point_to_segment( # 用遍历的方式计算每个障碍物圆心到新节点与最近节点之间的线段的最短距离，判断是否在障碍物内
                np.array([x1, y1]),
                np.array([x2, y2]),
                np.array([ox, oy]))
            if dd <= size ** 2:
                return False  # collision
        return True

    def get_final_course(self, lastIndex):
        path = [[self.goal.x, self.goal.y]] # 将终点加入路径中
        while self.node_list[lastIndex].parent is not None: # 通过父节点的索引，遍历解的路径上的节点，并放入path中
            node = self.node_list[lastIndex]
            path.append([node.x, node.y])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y])
        return path



    def draw_graph(self, rnd=None, path=None):
        plt.clf() # 使用matplotlib库pyplot模块中的clf()函数清除当前图形。
        # for stopping simulation with the esc key.
        # plt.gcf().canvas.mpl_connect( # gcf()用于获取当前图形
        #     'key_release_event',
        #     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k") # 在新节点处画一个^图形（上三角形），颜色为k（黑色）
            # matplotlib.pyplot.plot(*args, scalex=True, scaley=True, data=None, **kwargs)
            # x, y: 这些参数是数据点的水平和垂直坐标。X为可选值
            # fmt: 可选参数，包含字符串值
            # data: 可选参数，是带标签数据的对象。

        for node in self.node_list: # 通过遍历rrt上的点，画出一棵颜色为g（绿色）实线（-）的树
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x], [
                        node.y, self.node_list[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacle_list:   # 画出障碍物，颜色为k（黑色），形状为o（圆），半径为size
            # self.plot_circle(ox, oy, size)
            plt.plot(ox, oy, "ok", ms=20* size)  # 画一个圆形，颜色为k（黑色），线条粗细为30*size

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr") # 在终点和起点画颜色r（红色）的叉（x）

        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r') # 将解的路径用颜色r（红色）的实线（-）连起来

        plt.axis([-2, 18, -2, 18]) # 设置x轴-2到18，y轴-2到15
        plt.grid(True) # 设置是否配置网格线
        plt.pause(0.01) # 停止0.01秒


class Node:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


def main():
    print("Start rrt planning")

    # create obstacles
    # 障碍物列表
    obstacleList = [
        (3,  3,  1.5),
        (12, 2,  3),
        (3,  9,  2),
        (9,  11, 2),
    ]
    # obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
    #                 (9, 5, 2), (8, 10, 1)]

    # Set params
    rrt = RRT(randArea=[-2, 18], obstacleList=obstacleList, maxIter=200) # RRT类初始化，randArea为仿真空间，obstacleList为障碍物区域，maxIter为最大迭代次数
    path = rrt.rrt_planning(start=[0, 0], goal=[3, 9], animation=show_animation) # 调用rrt方法，获取从起点start到终点goal的路径，show_animation为是否在迭代时画出路线
    # path = rrt.rrt_star_planning(start=[0, 0], goal=[15, 12], animation=show_animation)
    # path = rrt.informed_rrt_star_planning(start=[0, 0], goal=[15, 12], animation=show_animation)
    print("Done!!")
    # print(path)
    if show_animation and path: # 程序结束后，是否继续显示图像
        plt.show()


if __name__ == '__main__':# 调用模块为本模块时调用main函数
    main()
