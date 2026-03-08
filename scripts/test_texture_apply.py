#!/usr/bin/env python3
"""Diagnose: verify connection works, then test texture streaming."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Sanity check 1: weather change (should be immediately visible)
print("=== TEST 1: Weather change ===")
weather = world.get_weather()
print(f"  Current sun altitude: {weather.sun_altitude_angle}")
weather.sun_altitude_angle = -90  # night time
weather.cloudiness = 100
world.set_weather(weather)
print("  Set to NIGHT + 100% clouds. Check editor — did it get dark?")
time.sleep(3)

# Restore
weather.sun_altitude_angle = 45
weather.cloudiness = 0
world.set_weather(weather)
print("  Restored to daytime.")
time.sleep(1)

# Sanity check 2: Can we read the world properly?
print("\n=== TEST 2: World info ===")
map_name = world.get_map().name
print(f"  Map: {map_name}")
settings = world.get_settings()
print(f"  Sync mode: {settings.synchronous_mode}")
print(f"  No rendering: {settings.no_rendering_mode}")

# Test 3: Try texture with sync mode enabled
print("\n=== TEST 3: Texture with sync mode ===")
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
settings.no_rendering_mode = False
world.apply_settings(settings)
world.tick()

size = 64
tex = carla.TextureColor(size, size)
for x in range(size):
    for y in range(size):
        tex.set(x, y, carla.Color(255, 255, 0, 255))

# Try road first
print("  Applying YELLOW to Road_Grass_Town01_37...")
world.apply_color_texture_to_object('Road_Grass_Town01_37', carla.MaterialParameter.Diffuse, tex)
for _ in range(30):
    world.tick()
time.sleep(2)
print("  Check editor — did any grass/road turn yellow?")

# Try tesla
print("  Applying YELLOW to SM_TeslaM3_parked_2...")
world.apply_color_texture_to_object('SM_TeslaM3_parked_2', carla.MaterialParameter.Diffuse, tex)
for _ in range(30):
    world.tick()
time.sleep(2)
print("  Check editor — did the car turn yellow?")

# Also try the float texture variant
print("\n=== TEST 4: Try apply_float_color_texture_to_object ===")
try:
    ftex = carla.TextureFloatColor(size, size)
    for x in range(size):
        for y in range(size):
            ftex.set(x, y, carla.FloatColor(1.0, 1.0, 0.0, 1.0))
    world.apply_float_color_texture_to_object('SM_TeslaM3_parked_2', carla.MaterialParameter.Diffuse, ftex)
    for _ in range(10):
        world.tick()
    print("  FloatColor applied OK")
except Exception as e:
    print(f"  FloatColor failed: {e}")

print("\nDone. Report what you see in the editor.")
