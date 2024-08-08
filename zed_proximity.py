import cv2
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def callback(point_cloud_msg):
    # Converte a nuvem de pontos para uma lista de pontos
    points_list = []
    for point in pc2.read_points(point_cloud_msg, skip_nans=True):
        points_list.append(point)
    
    rospy.loginfo(f"Recebido {len(points_list)} pontos")




def main():
    rospy.init_node('zed_point_cloud_listener', anonymous=True)
    rospy.Subscriber("/zed/point_cloud", PointCloud2, callback)

    rospy.spin()

if __name__ == "__main__":
    main()
