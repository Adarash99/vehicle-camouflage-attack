import cv2
import numpy as np
import time
from threading import Lock
from CarlaHandler import *

# Control panel parameters
button_width = 100
button_height = 30
button_margin = 20
start_x = 20
start_y = 20
panel_height = 150


class Visualizer:
   
    """
    A class to visualize the CARLA simulation environment and control various parameters.
    """
    def __init__(self, town='Town10HD'):
        self.cam_window = "Visualization Window"
        self.seg_window = "Segmentation Window"
        self.seg_car_window = "Segmented Car Window"
        self.control_window = "Control Panel"   
        self.handler = CarlaHandler(town=town)
        time.sleep(5)  # Allow time for initialization
        self.handler.spawn_vehicle('vehicle.tesla.model3')
        self.buttons = []
        self.panel = None
        # Initialize with black image
        self.image = np.zeros((400, 600, 3), dtype=np.uint8)
        self.segmentation = np.zeros((400, 600, 3), dtype=np.uint8)
        # Initialize the control panel
        self._display_control_window()
        # Intialize visualization window
        cv2.namedWindow(self.cam_window)
        cv2.namedWindow(self.seg_window)
        cv2.namedWindow(self.seg_car_window)

    def __del__(self):
        cv2.destroyAllWindows()
        if hasattr(self, 'handler'):
            del self.handler  # Ensure CARLA cleanup

    def _display_control_window(self):
        """
        Initialize the control panel with buttons and trackbars.
        """             
        # Create control panel UI
        self.panel = np.zeros((panel_height, 600, 3), dtype=np.uint8) + 220
        
        # Create buttons
        
        self.buttons = [
            self.create_button(self.panel, "Top view", start_x, start_y, button_width, button_height, self.top_view_callback),
            self.create_button(self.panel, "3d-view", start_x + button_width + button_margin, start_y, button_width, button_height, self.threed_view_callback)
        ]

        # Create window and UI elements

        len_spawn_points = self.handler.get_spawn_points()

        cv2.namedWindow(self.control_window)
        # Weather Trackbars
        cv2.createTrackbar("Cloudiness", self.control_window, 0, 100, self.cloudiness_callback)
        cv2.createTrackbar("Precipitation", self.control_window, 0, 100, self.precipitation_callback)
        cv2.createTrackbar("Deposits", self.control_window, 0, 100, self.precipitation_deposits_callback)
        cv2.createTrackbar("Wind", self.control_window, 0, 100, self.wind_intensity_callback)
        cv2.createTrackbar("Azimuth", self.control_window, 0, 360, self.sun_azimuth_angle_callback)
        cv2.createTrackbar("Altitude", self.control_window, 0, 180, self.sun_altitude_angle_callback)
        cv2.createTrackbar("Fog 1", self.control_window, 0, 100, self.fog_density_callback)
        cv2.createTrackbar("Fog 2", self.control_window, 0, 100, self.fog_distance_callback)
        cv2.createTrackbar("Spawn", self.control_window, 0, len_spawn_points, self.spawn_point_callback)
        cv2.createTrackbar("Color", self.control_window, 0, 180, self.color_callback)
        cv2.createTrackbar("Distance", self.control_window, 0, 100, self.distance_callback)
        cv2.createTrackbar("Pitch", self.control_window, 0, 90, self.pitch_callback)
        cv2.createTrackbar("Yaw", self.control_window, 0, 360, self.yaw_callback)
        
        cv2.setMouseCallback(self.control_window, self._mouse_callback, [self.panel])
        cv2.namedWindow(self.control_window)

    def run(self):
        """
        Main loop to run the visualization and update the control panel.
        """
        while True:
            self.image = self.handler.get_image()
            self.segmentation = self.handler.get_segmentation()
            self.segmented_car = self.handler.get_segmented_car()
            cv2.imshow(self.cam_window, self.image)
            cv2.imshow(self.seg_window, self.segmentation)
            cv2.imshow(self.seg_car_window, self.segmented_car)
            cv2.imshow(self.control_window, self.panel)
            self.handler.world.tick()  # Tick the world to update the simulation
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is in the panel area (below the image)
            if y >= param[0].shape[0] - panel_height:
                # Adjust y coordinate to panel coordinates
                panel_y = y - (param[0].shape[0] - panel_height)
                
                # Check if any button was clicked
                for (x1, y1, x2, y2, callback) in self.buttons:
                    if x1 <= x <= x2 and y1 <= panel_y <= y2:
                        callback()
                        break

    def _get_color(self, hue_value, saturation=255, value=255):
        """
        Convert HSV color to BGR for OpenCV.
        Returns:
            BGR color tuple for OpenCV.
        """
        # OpenCV uses H:0-180, S:0-255, V:0-255        
        hsv_color = np.uint8([[[hue_value, saturation, value]]])
        print(str(hsv_color))
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
        #print(rgb_color)
        return tuple(rgb_color[0][0])
    
    '''
    View Callbacks
    '''

    def top_view_callback(self):
        print("Top view clicked")
        self.handler.update_view('top')

    def threed_view_callback(self):
        print("3D view clicked")
        self.handler.update_view('3d')

    '''
    Weather Callbacks
    '''

    def cloudiness_callback(self, value):
        print(f"cloudiness value: {value}")
        self.handler.update_cloudiness(value)

    def precipitation_callback(self, value):
        print(f"precipitation value: {value}")
        self.handler.update_precipitation(value)

    def precipitation_deposits_callback(self, value): 
        print(f"precipitation deposits value: {value}")
        self.handler.update_precipitation_deposits(value)

    def wind_intensity_callback(self, value):
        print(f"wind intensity value: {value}")
        self.handler.update_wind_intensity(value)

    def sun_azimuth_angle_callback(self, value):
        print(f"sun azimuth angle value: {value}")
        self.handler.update_sun_azimuth_angle(value)

    def sun_altitude_angle_callback(self, value):
        value = 90 - value  # Invert the value for altitude angle
        print(f"sun altitude angle value: {value}")
        self.handler.update_sun_altitude_angle(value)

    def fog_density_callback(self, value):
        print(f"fog density value: {value}")
        self.handler.update_fog_density(value)

    def fog_distance_callback(self, value):
        print(f"fog distance value: {value}")
        self.handler.update_fog_distance(value)

    ''' 
    Other Callbacks 
    '''

    def spawn_point_callback(self, value):
        print(f"spawn value: {value}")
        self.handler.change_spawn_point(value)

    def color_callback(self, value):
        print(f"color value: {value}")
        color = self._get_color(value)
        self.handler.change_vehicle_color(color)

    def distance_callback(self, value):
        print(f"distance value: {value}")
        self.handler.update_distance(value)

    def pitch_callback(self, value):
        print(f"pitch value: {value}")
        self.handler.update_pitch(value)

    def yaw_callback(self, value):
        print(f"yaw value: {value}")
        self.handler.update_yaw(value)


    def create_button(self, panel, text, x, y, width, height, callback):
        # Draw button
        cv2.rectangle(panel, (x, y), (x + width, y + height), (200, 200, 200), -1)
        cv2.rectangle(panel, (x, y), (x + width, y + height), (50, 50, 50), 2)
        
        # Calculate text position to center it
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x + (width - text_width) // 2
        text_y = y + (height + text_height) // 2
        
        # Put text on button
        cv2.putText(panel, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Return button coordinates for click detection
        return (x, y, x + width, y + height, callback)
 
    
if __name__ == '__main__':

    while True:
        option = input("Enter the town (e.g., Town10HD) or enter 'exit': ")
        if option == '':
            option = 'Town10HD'
        if option== 'exit':
            print("Exiting...")
            break
        else:
            try:
                # Initialize the Visualizer with the specified town
                viz = Visualizer(option)
                time.sleep(5)  # Allow time for initialization
                viz.run() # Run the visualization loop
            finally:
                # Cleanup
                cv2.destroyAllWindows()
                del viz