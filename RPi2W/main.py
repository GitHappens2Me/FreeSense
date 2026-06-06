import bluetooth
import struct
import time

from machine import Pin, I2C

# Using https://github.com/dobodu/BOSCH-BNO085-I2C-micropython-library
from bno08x import *

# Pin-Layout
# RPI PICO 2 W          BNO085 BREAKOUT
# +-----------+          +-------------+
# |           |          |             |
# |  3V3 (36) |--------->|  VIN        |  <-- Power (3.3V)
# |           |          |             |
# |  GND (38) |--------->|  GND        |  <-- Ground
# |           |          |             |
# |  GP4 (6)  |--------->|  SDA        |  <-- I2C Data
# |           |          |             |
# |  GP5 (7)  |--------->|  SCL        |  <-- I2C Clock
# |           |          |             |
# +-----------+          +-------------+

# Maximum frequeny (Hz) of BLE packages, might be lower due to processing time
BLE_SEND_FREQUENCY = 20





# SDA & SCL using GPIO-4 and GPIO-5 (see diagram above)
i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=100000)
bno = BNO08X(i2c, address=0x4A)

# Only enable the sensor features we need
# Explanations at https://learn.adafruit.com/adafruit-9-dof-orientation-imu-fusion-breakout-bno085/report-types
# Where applicable, we set report frequency to the specified BLE frequency or the sensor's default if it is higher
# Defaults at: https://github.com/dobodu/BOSCH-BNO085-I2C-micropython-library/blob/b8253eb752cda5b113998ee66dd650217d3a7ab5/lib/bno08x.py#L165
# TODO(?): One could also set sensor frequency higher than BLE_SEND_FREQUENCY and average over multiple readings
bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION, freq=max(BLE_SEND_FREQUENCY, 20)) # Gravity-removed acceleration
bno.enable_feature(BNO_REPORT_ROTATION_VECTOR, freq=max(BLE_SEND_FREQUENCY, 10))     # Rotation using Accel + Gyro + Magnetometer
bno.set_quaternion_euler_vector(BNO_REPORT_ROTATION_VECTOR) # bno.euler now uses BNO_REPORT_ROTATION_VECTOR, default is BNO_REPORT_GAME_ROTATION_VECTOR

# ---
#bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)  # Rotation using Accel + Gyro (no magnetic interference possible)
#bno.enable_feature(BNO_REPORT_STABILITY_CLASSIFIER)  # Detect when stationary
# ---

time.sleep_ms(200)

# BLE (Bluetooth Low Energy)
ble = bluetooth.BLE()
ble.active(True)

UART_SERVICE = bluetooth.UUID("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
UART_TX = bluetooth.UUID("6E400003-B5A3-F393-E0A9-E50E24DCCA9E")

tx_char = (UART_TX, bluetooth.FLAG_NOTIFY)
service = (UART_SERVICE, [tx_char])
handles = ble.gatts_register_services([service])
tx_handle = handles[0][0]

connected = False
conn_handle = None

def ble_irq(event, data):
    global connected, conn_handle
    if event == 1:
        conn_handle, _, _ = data
        connected = True
        print("Connected!")
    elif event == 2:
        connected = False
        conn_handle = None
        print("Disconnected")
        ble.gap_advertise(100, b'\x02\x01\x06\x09\x09Freeskate')

ble.irq(ble_irq)
ble.gap_advertise(100, b'\x02\x01\x06\x09\x09Freeskate')
print("Advertising...")

time_between_sends_ms = 1000 // BLE_SEND_FREQUENCY

while True:
    timestamp_ms = time.ticks_ms()
    
    try:
        # TODO: Using quaternion and computing Euler in software later might be better
        # For Conversion, see bno08x.py Line 775
        qw, qx, qy, qz = bno.quaternion
        roll, pitch, yaw = bno.euler
        accel_x, accel_y, accel_z = bno.acc_linear 
        
        if connected and conn_handle:
            # Packet 1: Timestamp (T:)
            pkt1 = f"T:{timestamp_ms}"
            ble.gatts_notify(conn_handle, tx_handle, pkt1.encode())
            time.sleep_ms(3)
            
            # Packet 2: Angles (A:)
            pkt2 = f"A:{pitch:.1f},{yaw:.1f},{roll:.1f}"
            ble.gatts_notify(conn_handle, tx_handle, pkt2.encode())
            time.sleep_ms(3)
            
            # Packet 3: Acceleration (G:)
            pkt3 = f"G:{accel_x:.2f},{accel_y:.2f},{accel_z:.2f}"
            ble.gatts_notify(conn_handle, tx_handle, pkt3.encode())
        
        print(f"{timestamp_ms}: P={pitch:.1f} Y={yaw:.1f} R={roll:.1f} | A={accel_x:.2f},{accel_y:.2f},{accel_z:.2f} | Cal:{bno.calibration_status}")
        #print(f"Q:{qw},{qx},{qy},{qz} | {accel_x},{accel_y},{accel_z}")
    except Exception as e:
        print(f"Error: {e}")
        raise e
    
    elapsed_time_ms = time.ticks_diff(time.ticks_ms(), timestamp_ms)
    sleep_time_ms = time_between_sends_ms - elapsed_time_ms
    if sleep_time_ms > 0:
        #print(f"Target time: {time_between_sends_ms}ms,now sleeping: {sleep_time_ms}ms")
        time.sleep_ms(sleep_time_ms)