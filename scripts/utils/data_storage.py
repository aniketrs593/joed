import h5py
import numpy as np


class DataStorageWriter:

    def __init__(self, file_path, sensor_names=["lidar1", "camera_rgb1", "camera_semseg1", "gnss", "bouding_boxes"]):
        self.h5 = h5py.File(file_path, 'w')
        self.groups = dict()
        for group_name in sensor_names:
            self.groups[group_name] = self.h5.create_group(group_name)
    
    def write_lidar(self, sensor_name, index, lidar_measurement):
        x, y, z = list(), list(), list()
        for location in lidar_measurement:
            x.append(location.x)
            y.append(location.y)
            z.append(location.z)
        value = np.hstack([np.array(x).reshape((-1, 1)), np.array(y).reshape((-1, 1)), np.array(z).reshape((-1, 1))])
        self.write_matrix(sensor_name, index, value)
    
    def write_gnss(self, sensor_name, index, gnss_data):
        x, y, z = list(), list(), list()
        x.append(gnss_data.latitude)
        y.append(gnss_data.longitude)
        z.append(gnss_data.altitude)
        value = np.hstack([np.array(x).reshape((-1, 1)), np.array(y).reshape((-1, 1)), np.array(z).reshape((-1, 1))])
        self.write_matrix(sensor_name, index, value)
        
    def write_image(self, sensor_name, index, image):
        self.write_matrix(sensor_name, index, image)

    def write_collision(self, sensor_name, index, collision_event):
        self.write_matrix(sensor_name, index,
                          np.array([collision_event.normal_impulse.x,
                                    collision_event.normal_impulse.y,
                                    collision_event.normal_impulse.z]).reshape((1, -1)))
    
    def write_matrix(self, sensor_name, index, value):
        self.groups[sensor_name].create_dataset(str(index), data=value, maxshape=value.shape, chunks=True)
    
    def set_attributes(self, sensor_name, attrs):
        for param in attrs:
            self.groups[sensor_name].attrs[param] = attrs[param]
    
    def close(self):
        self.h5.close()
        self.h5 = None
