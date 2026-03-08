import carla
import numpy as np
import cv2
import math
from threading import Lock
import random


class CarlaHandler():

    spectator_views = ['top', '3d']
   
  
    def __init__(self, host='localhost', port=2000, town='Town10HD', x_res=1024, y_res=1024, no_rendering=True, skip_load_world=False):

        """
        Initialize the CARLA client, world and set the world settings.

        Args:
            no_rendering: If True (default), disables rendering for performance during
                         training. Set to False for visualization/texture application.
            skip_load_world: If True, connect to the existing world without reloading.
                           Use this when connecting to a UE4 editor session.
        """
        # Connect to CARLA server
        self.client = carla.Client(host, port)
        self.client.set_timeout(30.0)  # Set timeout for client requests
        if not skip_load_world:
            self.client.load_world(town)
        self.world = self.client.get_world()
        self.weather = self.world.get_weather()

        # Configure CARLA world settings
        settings = self.world.get_settings()
        settings.no_rendering_mode = no_rendering
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.x_res = x_res  
        self.y_res = y_res

        self.spectator = self.world.get_spectator()
        self.blueprint_library = self.world.get_blueprint_library()

        # Add a camera sensor to the spectator
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.x_res))
        camera_bp.set_attribute('image_size_y', str(self.y_res))
        camera_bp.set_attribute('sensor_tick', '0.1')
        self.camera = self.world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location()),
            attach_to=self.spectator
        )
        self.camera.listen(self.camera_callback)

        # Add segmentation camera to the spectator
        seg_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(self.x_res))
        seg_bp.set_attribute('image_size_y', str(self.y_res))
        seg_bp.set_attribute('sensor_tick', '0.1')
        self.seg_camera = self.world.spawn_actor(
            seg_bp,
            carla.Transform(carla.Location()),
            attach_to=self.spectator
        )
        self.seg_camera.listen(self.seg_camera_callback)

        # Add depth camera to the spectator
        depth_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(self.x_res))
        depth_bp.set_attribute('image_size_y', str(self.y_res))
        depth_bp.set_attribute('sensor_tick', '0.1')
        self.depth_camera = self.world.spawn_actor(
            depth_bp,
            carla.Transform(carla.Location()),
            attach_to=self.spectator
        )
        self.depth_data = None
        self.depth_lock = Lock()
        self.depth_camera.listen(self.depth_camera_callback)

        # Spectator settings
        self.spectator_view = 'top'
        self.spectator_pitch = 15
        self.spectator_yaw = 0
        self.spectator_distance = 10

        # Weather settings
        self.weather.cloudiness = 0.0
        self.weather.precipitation = 0.0
        self.weather.precipitation_deposits = 0.0
        self.weather.wind_intensity = 0.0
        self.weather.sun_azimuth_angle = 0.0
        self.weather.sun_altitude_angle = 15.0
        self.weather.fog_density = 0.0  
        self.weather.fog_distance = 0.0
        self.world.set_weather(self.weather)

        # Other settings
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = None
        self.spawn_point = None
        self.spawn_point_index = 0
        self.image = None
        self.seg_data = {'mask': None}
        self.seg_labels = None
        self.image_lock = Lock()
        self.seg_lock = Lock()

      
    def  __del__(self):
        """
        Cleanup function to destroy the vehicle and camera when the object is deleted.
        """
        print('called the __del__ function')

        try:

            if self.vehicle is not None:
                self.vehicle.destroy()
            if self.camera is not None:
                self.camera.destroy()
            if self.seg_camera is not None:
                self.seg_camera.destroy()
            if self.depth_camera is not None:
                self.depth_camera.destroy()
            if self.spectator is not None:
                self.spectator.destroy()
            self.world.apply_settings(carla.WorldSettings())  # Reset to default settings
        finally:
            print('Ending. . .')

    def _update_spectator(self):
        
        if self.spectator_view == 'top':
            # Position spectator above the car
            camera_transform = carla.Transform(self.spawn_point.location + carla.Location(z=self.spectator_distance), 
                carla.Rotation(pitch=-90, yaw=self.spectator_yaw)  # Look straight down    
            )
        elif self.spectator_view == '3d':
            # Position spectator at a 3D view angle
            vehicle_yaw = self.spawn_point.rotation.yaw
            vehicle_location = self.spawn_point.location
            # Convert angles to radians for math functions
            yaw_rad = math.radians(vehicle_yaw + self.spectator_yaw)
            pitch_rad = math.radians(self.spectator_pitch)

            # Calculate camera offset from vehicle
            x_offset = math.cos(yaw_rad) * math.cos(pitch_rad) * self.spectator_distance
            y_offset = math.sin(yaw_rad) * math.cos(pitch_rad) * self.spectator_distance
            z_offset = math.sin(pitch_rad) * self.spectator_distance

            # Camera position (relative to vehicle)
            camera_pos = carla.Location(
            x=vehicle_location.x - x_offset,
            y=vehicle_location.y - y_offset,
            z=vehicle_location.z + z_offset
            )

            # Camera rotation (looking at the vehicle)
            camera_rot = carla.Rotation(
                pitch=-self.spectator_pitch,  # Tilt down toward the vehicle
                yaw=vehicle_yaw + self.spectator_yaw,
                roll=0
            )
            camera_transform = carla.Transform(camera_pos, camera_rot)

        self.spectator.set_transform(camera_transform)


    def camera_callback(self, carla_image):
        # Convert CARLA image to OpenCV format
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        #array = array[:, :, ::-1]  # Convert BGR to RGB
        with self.image_lock:
            # Resize the camera frame to match the display dimensions
            #array = cv2.resize(array, (600, 400))
            self.image = array.copy() if array is not None else None
            

    def seg_camera_callback(self, image):
        """
        Convert segmentation to show only cars in blue (others black)
        """
        # Get raw segmentation data (CityScapes labels)
        image.convert(carla.ColorConverter.Raw)  # Get raw labels first
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.seg_labels = array[:, :, 2]  # Semantic labels are in channel 2

        # Create blank black image (BGR format)
        mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        
        # Set car pixels to blue (BGR format)
        mask[np.isin(self.seg_labels, [14])] = [255, 0, 0]  # 14 = car
        
        with self.seg_lock:
            self.seg_data['mask'] = mask

    


    def depth_camera_callback(self, image):
        """Decode CARLA depth image to meters."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # CARLA depth encoding: depth = 1000 * (R + G*256 + B*65536) / (2^24 - 1)
        r = array[:, :, 2].astype(np.float32)
        g = array[:, :, 1].astype(np.float32)
        b = array[:, :, 0].astype(np.float32)
        depth_meters = 1000.0 * (r + g * 256.0 + b * 65536.0) / (16777215.0)
        with self.depth_lock:
            self.depth_data = depth_meters

    def get_depth(self):
        """Get depth map in meters.

        Returns:
            numpy [H, W] float32, depth in meters per pixel
        """
        with self.depth_lock:
            if self.depth_data is None:
                return np.zeros((self.y_res, self.x_res), dtype=np.float32)
            return self.depth_data.copy()

    def get_camera_transform(self):
        """Get the current spectator/camera transform.

        Returns:
            carla.Transform with camera position and rotation
        """
        return self.spectator.get_transform()

    def get_vehicle_transform(self):
        """Get the vehicle's current transform.

        Returns:
            carla.Transform with vehicle position and rotation
        """
        if self.vehicle is None:
            raise ValueError("No vehicle spawned. Call spawn_vehicle() first.")
        return self.vehicle.get_transform()

    def get_image(self):
        with self.image_lock:
            if self.image is None:
                # If image is None, return a black image of the same size
                print("Image is None")
                return np.zeros((self.x_res, self.y_res, 3), dtype=np.uint8)
            return self.image.copy() if self.image is not None else None
        
    def get_segmentation(self):
        with self.seg_lock:
            if self.seg_data['mask'] is None:
                # If segmentation data is None, return a black image of the same size
                print("Segmentation data is None")
                return np.zeros((self.x_res, self.y_res, 3), dtype=np.uint8)
            return self.seg_data['mask'].copy() if self.seg_data['mask'] is not None else None
        
    def get_segmented_car(self):
        """Extract only car pixels from RGB using segmentation mask"""
        with self.image_lock and self.seg_lock:
            if self.image is None or self.seg_data is None:
                return np.zeros((self.x_res, self.y_res, 3), dtype=np.uint8)
            
            # Create mask for vehicles (classes 13-18)
            vehicle_mask = np.isin(self.seg_labels, [14])  # 14 = car
            
            # Extract car pixels
            car_only = np.zeros_like(self.image)
            car_only[vehicle_mask] = self.image[vehicle_mask]
            
            return car_only
        
    def destroy_all_vehicles(self):
        """
        Safely destroy all vehicles in the CARLA world
        """
        # Get all actors and filter for vehicles
        vehicle_list = self.world.get_actors().filter('vehicle.*')
        
        # Batch destroy command (most efficient method)
        destroy_commands = [carla.command.DestroyActor(vehicle) for vehicle in vehicle_list]
        self.client.apply_batch(destroy_commands)

    def spawn_vehicle(self, model, color=(255, 0, 0)):

        vehicle_bp = self.blueprint_library.find(model)
        vehicle_bp.set_attribute('color', f'{color[0]}, {color[1]}, {color[2]}')
        self.spawn_point = self.spawn_points[self.spawn_point_index]
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)
        # Position spectator above the car
        self._update_spectator()


    def change_spawn_point(self, index=random.randint(0, 30)):
        """
        Change the spawn point of the vehicle to a new location.
        """
        if 0 <= index < len(self.spawn_points):
            self.spawn_point_index = index
            self.spawn_point = self.spawn_points[self.spawn_point_index]
            self.vehicle.set_transform(self.spawn_point)
            self._update_spectator()
        else:
            print("Invalid spawn point index")

    def change_vehicle_color(self, color):
        """
        Change the color of the vehicle.
        """
        if self.vehicle is not None:
            self.vehicle.destroy()
            vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
            chosen_color = str(color[0]) + ', ' + str(color[1]) + ', ' + str(color[2]) # Convert to string
            vehicle_bp.set_attribute('color', str(chosen_color))
            self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

    def world_tick(self, value=1):
        """
        Advance the world simulation by 'value' number of ticks.
        """
        for _ in range(value):
            # Advance the world simulation by one tick
            self.world.tick()
        
    def get_spawn_points(self):
        """
        Get the list of spawn points in the world.
        """
        return len(self.spawn_points)
    
    def update_view(self, option):
        """
        Update the spectator view based on the selected option.
        """
        if option in self.spectator_views:
            self.spectator_view = option
            self._update_spectator()
        else:
            print("Invalid spectator option. Choose 'top' or '3d'.")

    def update_distance(self, distance):
        """
        Update the distance of the spectator from the vehicle.
        """
        self.spectator_distance = distance
        self._update_spectator()

    def update_pitch(self, pitch):
        """
        Update the pitch angle of the spectator camera.
        """
        self.spectator_pitch = pitch
        self._update_spectator()

    def update_yaw(self, yaw):  
        """
        Update the yaw angle of the spectator camera.
        """
        self.spectator_yaw = yaw
        self._update_spectator()
    
    '''
    Weather functions
    '''
    def update_cloudiness(self, value):
        """
        Update the cloudiness of the weather.
        """
        self.weather.cloudiness = value
        self.world.set_weather(self.weather)

    def update_precipitation(self, value):
        """
        Update the precipitation of the weather.
        """
        self.weather.precipitation = value
        self.world.set_weather(self.weather)

    def update_precipitation_deposits(self, value):
        """
        Update the precipitation deposits of the weather.
        """
        self.weather.precipitation_deposits = value
        self.world.set_weather(self.weather)

    def update_wind_intensity(self, value):
        """
        Update the wind intensity of the weather.
        """
        self.weather.wind_intensity = value
        self.world.set_weather(self.weather)

    
    def update_sun_azimuth_angle(self, value):
        """
        Update the azimuth of the sun.
        """
        self.weather.sun_azimuth_angle = value
        self.world.set_weather(self.weather)

    def update_sun_altitude_angle(self, value):
        """
        Update the altitude of the sun.
        """
        self.weather.sun_altitude_angle = value
        self.world.set_weather(self.weather)
    
    def update_fog_density(self, value):
        """
        Update the fog density of the weather.
        """
        self.weather.fog_density = value
        self.world.set_weather(self.weather)

    def update_fog_distance(self, value):
        """
        Update the fog distance of the weather.
        """
        self.weather.fog_distance = value
        self.world.set_weather(self.weather)

    def set_camera_viewpoint(self, yaw, pitch, distance):
        """
        Position camera at specified viewpoint relative to vehicle.

        For EOT training: captures vehicle from different angles while keeping
        vehicle stationary. Camera orbits around vehicle at fixed distance.

        Args:
            yaw: Horizontal angle in degrees (0=front, 90=right, 180=back, 270=left)
            pitch: Vertical angle in degrees (negative = looking down at vehicle)
            distance: Distance from vehicle center in meters

        Example:
            handler.set_camera_viewpoint(yaw=60, pitch=-15, distance=8)
            # Camera positioned 60° clockwise from front, looking down 15°, 8m away
        """
        if self.vehicle is None:
            raise ValueError("No vehicle spawned. Call spawn_vehicle() first.")

        # Get vehicle transform
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = vehicle_transform.rotation.yaw

        # Convert angles to radians
        yaw_rad = math.radians(vehicle_yaw + yaw)
        pitch_rad = math.radians(pitch)

        # Calculate camera position offset from vehicle
        x_offset = math.cos(yaw_rad) * math.cos(pitch_rad) * distance
        y_offset = math.sin(yaw_rad) * math.cos(pitch_rad) * distance
        z_offset = math.sin(pitch_rad) * distance

        # Camera position (orbiting around vehicle)
        camera_pos = carla.Location(
            x=vehicle_location.x - x_offset,
            y=vehicle_location.y - y_offset,
            z=vehicle_location.z + z_offset + 1.5  # +1.5m to account for vehicle height
        )

        # Camera rotation (always looking at vehicle center)
        camera_rot = carla.Rotation(
            pitch=-pitch,  # Negative because CARLA pitch convention
            yaw=vehicle_yaw + yaw + 180,  # +180 to look back at vehicle
            roll=0
        )

        # Update spectator camera
        camera_transform = carla.Transform(camera_pos, camera_rot)
        self.spectator.set_transform(camera_transform)

        # Tick world to apply changes
        self.world.tick()

    def get_car_segmentation_mask(self):
        """Get binary car segmentation mask from CARLA semantic segmentation.

        Returns:
            numpy [H, W] float32 in [0, 1], 1.0 = car pixel
        """
        if self.seg_labels is None:
            return np.zeros((self.y_res, self.x_res), dtype=np.float32)
        return (self.seg_labels == 14).astype(np.float32)

    def get_paintable_mask(self, color1=(128, 128, 128), color2=(255, 0, 0), threshold=0.02):
        """
        Generate mask showing only paintable surfaces.

        Renders car in two different colors and compares - pixels that change
        between the two renders are paintable surfaces (body panels). Pixels that
        remain the same are structural features (rims, windows, headlights, plates).

        Args:
            color1: First vehicle color (R, G, B), default gray
            color2: Second vehicle color (R, G, B), default red
            threshold: Minimum normalized difference to consider a pixel as paintable.
                      Higher values = stricter (fewer pixels marked paintable).
                      Default: 0.02

        Returns:
            paintable_mask: Binary mask (H, W), float32, 1.0 = paintable, 0.0 = preserve
        """
        import time

        # Capture with first color
        self.change_vehicle_color(color1)
        self.world_tick(50)
        time.sleep(0.1)  # Wait for render to stabilize
        ref1 = self.get_image().astype(np.float32) / 255.0

        # Capture with second color
        self.change_vehicle_color(color2)
        self.world_tick(50)
        time.sleep(0.1)
        ref2 = self.get_image().astype(np.float32) / 255.0

        # Get CARLA car segmentation mask
        seg = self.get_segmentation()
        car_mask = (seg[:, :, 0] == 255) & (seg[:, :, 1] == 0) & (seg[:, :, 2] == 0)

        # Find pixels that changed significantly (paintable surfaces)
        diff = np.abs(ref1 - ref2).max(axis=-1)  # Max diff across RGB channels
        paintable = diff > threshold

        # Combine: must be car AND paintable
        paintable_mask = car_mask & paintable

        return paintable_mask.astype(np.float32)

    def apply_texture_to_vehicle(self, texture_image, object_name=None):
        """Apply a texture image to the vehicle body via CARLA's texture streaming API.

        Requires no_rendering=False to be visible in captures.

        Args:
            texture_image: numpy HxWx3 uint8 RGB image to apply as diffuse texture
            object_name: CARLA object name for the vehicle mesh. If None, attempts
                        auto-discovery by probing common patterns.

        Returns:
            str: The object name that was successfully applied, or None if failed.
        """
        if self.vehicle is None:
            raise ValueError("No vehicle spawned. Call spawn_vehicle() first.")

        h, w = texture_image.shape[:2]

        # Build carla.TextureColor (Y-flipped per CARLA convention)
        tex = carla.TextureColor(w, h)
        for x in range(w):
            for y in range(h):
                r, g, b = int(texture_image[y, x, 0]), int(texture_image[y, x, 1]), int(texture_image[y, x, 2])
                tex.set(x, h - y - 1, carla.Color(r, g, b, 255))

        if object_name is not None:
            # Apply directly to the specified name
            self.world.apply_color_texture_to_object(
                object_name, carla.MaterialParameter.Diffuse, tex
            )
            return object_name

        # Auto-discover: try common patterns
        all_names = self.world.get_names_of_all_objects()
        vehicle_keywords = ['tesla', 'vehicle', 'car', 'body', 'paint', 'chassis', 'sedan']
        candidates = [n for n in all_names
                      if any(kw in n.lower() for kw in vehicle_keywords)]

        # Capture baseline for comparison
        self.world_tick(5)
        baseline = self.get_image()

        for name in candidates:
            try:
                self.world.apply_color_texture_to_object(
                    name, carla.MaterialParameter.Diffuse, tex
                )
                self.world_tick(5)
                after = self.get_image()
                diff = np.abs(baseline.astype(float) - after.astype(float)).mean()
                if diff > 5.0:
                    return name
            except Exception:
                continue

        return None