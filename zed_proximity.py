#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np

def callback(data):
    # Converte a nuvem de pontos para um array numpy
    pc_data = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(pc_data))

    if points.size > 0:
        # Calcula a distância para cada ponto e encontra o ponto mais próximo
        distances = np.linalg.norm(points, axis=1)
        min_distance_index = np.argmin(distances)
        closest_point = points[min_distance_index]

        print(f"Ponto mais próximo: {closest_point}")
        print(f"Distância do ponto mais próximo: {distances[min_distance_index]} m")

def listener():
    rospy.init_node('point_cloud_listener', anonymous=True)
    rospy.Subscriber('/zed/point_cloud', PointCloud2, callback)
    rospy.spin()
    
def 

if __name__ == '__main__':
    listener()
