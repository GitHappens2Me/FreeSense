import bluetooth
import struct
from machine import Pin, I2C
from bno08x import *
import time

# Setup
i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=100000)
bno = BNO08X(i2c, address=0x4A)
bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)

# BLE
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

last_send = time.ticks_ms()

while True:
    current_time = time.ticks_ms()
    if time.ticks_diff(current_time, last_send) < 50:
        continue
    last_send = current_time
    
    try:
        roll, pitch, yaw = bno.euler
        accel_x, accel_y, accel_z = bno.acc
        
        if connected and conn_handle:
            # Packet 1: Timestamp (T:)
            pkt1 = f"T:{current_time}"
            ble.gatts_notify(conn_handle, tx_handle, pkt1.encode())
            time.sleep_ms(5)
            
            # Packet 2: Angles (A:)
            pkt2 = f"A:{pitch:.1f},{yaw:.1f},{roll:.1f}"
            ble.gatts_notify(conn_handle, tx_handle, pkt2.encode())
            time.sleep_ms(5)
            
            # Packet 3: Acceleration (G:)
            pkt3 = f"G:{accel_x:.2f},{accel_y:.2f},{accel_z:.2f}"
            ble.gatts_notify(conn_handle, tx_handle, pkt3.encode())
        
        print(f"{current_time}: P={pitch:.1f} Y={yaw:.1f} R={roll:.1f} | A={accel_x:.2f},{accel_y:.2f},{accel_z:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")