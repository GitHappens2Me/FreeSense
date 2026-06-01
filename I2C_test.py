from machine import Pin, I2C
import time

# Initialize I2C on your connected pins
i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=100000)

print("Scanning I2C bus...")
devices = i2c.scan()

if devices:
    print(f"Device(s) found at: {[hex(d) for d in devices]}")
    if 0x4a in devices:
        print("✓ BNO085 detected at address 0x4A")
    elif 0x4b in devices:
        print("✓ BNO085 detected at address 0x4B")
else:
    print("No I2C devices found - check wiring")