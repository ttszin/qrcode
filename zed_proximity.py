import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def callback(point_cloud_msg):
    # Processar a nuvem de pontos
    closest_point = find_closest_point(point_cloud_msg)
    if closest_point:
        rospy.loginfo(f"Ponto mais próximo: {closest_point}")
    else:
        rospy.loginfo("Nenhum ponto encontrado.")

def find_closest_point(point_cloud_msg):
    closest_point = None
    min_distance = float('inf')
    
    for point in pc2.read_points(point_cloud_msg, skip_nans=True):
        x, y, z = point[:3]
        distance = (x**2 + y**2 + z**2)**0.5
        
        if distance < min_distance:
            min_distance = distance
            closest_point = (x, y, z)
    
    return closest_point

def main():
    rospy.init_node('point_cloud_listener')
    rospy.Subscriber('/camera/depth/points', PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    main()
