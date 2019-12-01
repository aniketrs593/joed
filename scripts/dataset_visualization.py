

# Use Open3D to read the point cloud files and pyqtgraph to show
# OpenCV to read and draw images

# Use PyQt to select and display

# timeline to view the data over time


from utils.data_storage import DataStorageReader
import cv2



input_file_path = "../dataset/data_usecase2.h5"


reader = DataStorageReader(input_file_path)

print('> Datasets available:')
print('>> ', reader.sensor_labels)

indexes = reader.get_timestamps_as_string()

print(indexes)

for idx in indexes:
    img = reader.get_image(reader.rgb_cameras[-1], idx)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break


    