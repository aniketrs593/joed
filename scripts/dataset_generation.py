
from utils.data_storage import DataStorageWriter
from utils.carla_utils import *

import random
import time
import cv2

from tqdm import tqdm

sys.path.append("../simulator/carla/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg")

import carla


print("> Connecting to localhost:2000")
client = carla.Client('localhost', 2000)
client.set_timeout(4.0)

world = client.get_world()

print("> Configuring weather conditions")
weather = world.get_weather()
weather.cloudyness = 0
weather.precipitation = 0
weather.precipitation_deposits = 0
weather.wind_intensity = 0
weather.sun_azimuth_angle = 30
weather.sun_altitude_angle = 100
world.set_weather(weather)

blueprint_library = world.get_blueprint_library()
actor_list = []

###############################################################################################
###############################################################################################
# Ego and Vehicles creation

print("> Creating Vehicles")

print(">> Creating EGO vehicle")

bp = random.choice(blueprint_library.filter('vehicle.tesla.*'))
transform = carla.Transform(carla.Location(x=-88.5, y=-160, z=0.8), carla.Rotation(pitch=0.000000, yaw=90.0, roll=0.000000))
ego_vehicle = world.spawn_actor(bp, transform)
actor_list.append(ego_vehicle)

print(">> Creating Opponent 1")

bp = random.choice(blueprint_library.filter('vehicle.tesla.*'))
transform = carla.Transform(carla.Location(x=-88.5, y=-120, z=0.8), carla.Rotation(pitch=0.000000, yaw=90.0, roll=0.000000))
op_vehicle1 = world.spawn_actor(bp, transform)
actor_list.append(op_vehicle1)

###############################################################################################
###############################################################################################
# Sensor Definition

print("> Defining Sensors")

camera1_parameters = {"x": 0.0, "y": 0, "z": 1.5, "roll": 0, "pitch": -10, "yaw": 0,
                      "width": 1024, "height": 768, "fov": 45,
                      "sensor_label": "camera1",
                      "sensor_type": "camera"}

camera2_parameters = {"x": 0.0, "y": 0, "z": 1.5, "roll": 0, "pitch": -10, "yaw": 45,
                      "width": 1024, "height": 768, "fov": 45,
                      "sensor_label": "camera2",
                      "sensor_type": "camera"}

camera3_parameters = {"x": 0.0, "y": 0, "z": 1.5, "roll": 0, "pitch": -10, "yaw": 90,
                      "width": 1024, "height": 768, "fov": 45,
                      "sensor_label": "camera3",
                      "sensor_type": "camera"}

camera4_parameters = {"x": 0.0, "y": 0, "z": 1.5, "roll": 0, "pitch": -10, "yaw": 135,
                      "width": 1024, "height": 768, "fov": 45,
                      "sensor_label": "camera4",
                      "sensor_type": "camera"}

camera5_parameters = {"x": 0.0, "y": 0, "z": 1.5, "roll": 0, "pitch": -10, "yaw": 180,
                      "width": 1024, "height": 768, "fov": 45,
                      "sensor_label": "camera5",
                      "sensor_type": "camera"}

camera6_parameters = {"x": 0.0, "y": 0, "z": 1.5, "roll": 0, "pitch": -10, "yaw": 225,
                      "width": 1024, "height": 768, "fov": 45,
                      "sensor_label": "camera6",
                      "sensor_type": "camera"}

camera7_parameters = {"x": 0.0, "y": 0, "z": 1.5, "roll": 0, "pitch": -10, "yaw": 270,
                      "width": 1024, "height": 768, "fov": 45,
                      "sensor_label": "camera7",
                      "sensor_type": "camera"}

camera8_parameters = {"x": 0.0, "y": 0, "z": 1.5, "roll": 0, "pitch": -10, "yaw": 315,
                      "width": 1024, "height": 768, "fov": 45,
                      "sensor_label": "camera8",
                      "sensor_type": "camera"}

lidar1_parameters = {"x": 1.5, "y": 0, "z": 2.4, "roll": 0, "pitch": 0, "yaw": 0,
                     "channels": 32, "range": 100, "lower_fov": -30, "upper_fov": 15,
                     "sensor_label": "lidar",
                     "sensor_type": "camera"}

gnss_parameters = {"x": 1.5, "y": 0, "z": 2.4, "roll": 0, "pitch": 0, "yaw": 0,
                   "sensor_label": "gnss",
                    "sensor_type": "gnss"}

collision_parameters = {"sensor_label": "collision", "sensor_type": "collision"}


sensor_labels = ['camera1_rgb','camera1_semseg','camera1_depth',
                 'camera2_rgb','camera2_semseg','camera2_depth',
                 'camera3_rgb','camera3_semseg','camera3_depth',
                 'camera4_rgb','camera4_semseg','camera4_depth',
                 'camera5_rgb','camera5_semseg','camera5_depth',
                 'camera6_rgb','camera6_semseg','camera6_depth',
                 'camera7_rgb','camera7_semseg','camera7_depth',
                 'camera8_rgb','camera8_semseg','camera8_depth',
                 'lidar',
                 'gnss',
                 'collision']


# imu_parameters = {"x": 1.5, "y": 0, "z": 2.4, "roll": 0, "pitch": 0, "yaw": 0,
#                   "sensor_label": "imu",
#                   "sensor_type": "imu"}

###############################################################################################
###############################################################################################
# Sensor Creation
print("> Creating Sensors")

sensor_actor_list = []
sensors_parameters = []

print(">> Creating Camera Sensors")
for camera_parameter in [camera1_parameters, camera2_parameters, camera3_parameters, camera4_parameters, camera5_parameters, camera6_parameters, camera7_parameters, camera8_parameters]:
#for camera_parameter in [camera1_parameters, camera2_parameters]:
    sensor_actor_list += create_camera_sensors(world, ego_vehicle, camera_parameter)
    sensors_parameters.append(camera_parameter)

print(">> Creating LIDAR Sensors")
for lidar_parameter in [lidar1_parameters]:
    sensor_actor_list.append(create_lidar_sensor(world, ego_vehicle, lidar_parameter))
    sensors_parameters.append(lidar_parameter)

print(">> Creating GNSS Sensors")
for gnss_parameter in [gnss_parameters]:
    sensor_actor_list.append(create_gnss_sensor(world, ego_vehicle, gnss_parameter))
    sensors_parameters.append(gnss_parameter)

#print(">> Creating Collision Sensors")
#sensor_actor_list.append(world.spawn_actor(blueprint_library.find('sensor.other.collision'), carla.Transform(), attach_to=ego_vehicle))
#sensors_parameters.append({"sensor_label": "collision", "sensor_type": "collision"})

# # create imu sensor
# sensor_actor_list.append(create_imu_sensor(world, ego_vehicle, imu_parameters))
# sensors_parameters.append(imu_parameters)

# get name parameters list to create data storage file
sensor_labels = [param["sensor_label"] for param in sensors_parameters]


## Create Database

try:

    time.sleep(3)

    ds_file_path = "../dataset/data3.h5"
    print("> Creating data storage file", ds_file_path)
    ds = DataStorageWriter(ds_file_path, ["lidar", "camera", "gnss", "semantic"])
    
    print("> Starting simulation")
    vehicles = world.get_actors().filter('vehicle.*')    
    with CarlaSyncMode(world, 
                       sensor_actor_list,
                       fps=30) as sync_mode:
        
        ego_vehicle.set_velocity(carla.Vector3D(0.0, 10.3, 0))
        op_vehicle1.set_velocity(carla.Vector3D(0.0, 5.3, 0))
        
        for idx in tqdm(range(30*10)):
            data = sync_mode.tick(timeout=4.0)
            
            snapshot, image_rgb, image_seg, image_depth, lidar_pcl, gnss_data = data[:6]
                        
            image_seg.convert(carla.ColorConverter.CityScapesPalette)
            np_image = compute_data_buffer(image_rgb)
            
            bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, sensor_actor_list[1])
            np_image2 = ClientSideBoundingBoxes.draw_bounding_boxes(np_image, bounding_boxes)
            frame = snapshot.frame
            ts = int(snapshot.timestamp.elapsed_seconds * 1e6)
            
#             ds.write_lidar("lidar", ts, lidar_pcl)
#             ds.write_image("camera", ts,  compute_data_buffer(image_rgb))
#             ds.write_image("semantic", ts,  compute_data_buffer(image_seg))
#             ds.write_gnss("gnss", ts, gnss_data)
            
            imgs = []
            for idx in range(8):
                imgs.append(cv2.resize(compute_data_buffer(data[idx*3+1]), None, fx=0.25, fy=0.25))

            img1 = np.concatenate((imgs[5], imgs[6], imgs[7], imgs[0], imgs[1], imgs[2], imgs[3], imgs[4]), axis=1)
            cv2.imshow("img", img1)
            
            if cv2.waitKey(1) == ord('q'):
                break
            if ego_vehicle.get_location().y > 140:
                break
finally:
    
    print("> Cleaning Simulation")
    actor_list += sensor_actor_list
    
    for actor in actor_list:
        actor.destroy()
    actor_list = []
    cv2.destroyAllWindows()
    ds.close()

