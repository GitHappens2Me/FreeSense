# This code, when run on the raspberry Pi tests the I2C connection

from machine import Pin, I2C
import time
from bno08x import *  # All reports included here

# Initialize I2C
i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=100000)

# Create sensor object
bno = BNO08X(i2c, address=0x4A)

# Enable sensors
bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GYROSCOPE)
bno.enable_feature(BNO_REPORT_MAGNETOMETER)
bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)

print("Sensor ready! Reading data...\n")

while True:
    accel_x, accel_y, accel_z = bno.acc
    print(f"Accel: X={accel_x:+.3f} Y={accel_y:+.3f} Z={accel_z:+.3f} m/s²")
    
    gyro_x, gyro_y, gyro_z = bno.gyro
    print(f"Gyro:  X={gyro_x:+.3f} Y={gyro_y:+.3f} Z={gyro_z:+.3f} rad/s")
    
    mag_x, mag_y, mag_z = bno.mag
    print(f"Mag:   X={mag_x:+.3f} Y={mag_y:+.3f} Z={mag_z:+.3f} µT")
    
    roll, pitch, yaw = bno.euler

    print(f"Roll:  {roll:.1f}°")   # Rotation around X (tilting side to side)
    print(f"Pitch: {pitch:.1f}°")  # Rotation around Y (tilting forward/back)
    print(f"Yaw:   {yaw:.1f}°")    # Rotation around Z (compass heading)
    
    print("-" * 40)
    time.sleep(0.5)