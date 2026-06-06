import time
from machine import Pin, I2C
from bno08x import *

i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=100000)
bno = BNO08X(i2c, address=0x4A)

print("Starting magnetometer calibration...")

# Enable magnetometer first
bno.enable_feature(BNO_REPORT_MAGNETOMETER)
time.sleep_ms(200)
mx, my, mz = bno.mag
print(f"Raw mag: {mx}, {my}, {mz}")

if mx == 0.0 and my == 0.0 and mz == 0.0:
    print("Magnetometer is DEAD or blocked")
else:
    print(f"Field strength: {((mx**2 + my**2 + mz**2)**0.5):.1f} uT")
    print("Earth field is ~25-65 uT. If much higher, you're near interference!")

bno.calibration  # Start calibration mode
time.sleep_ms(100)

print("\n" + "=" * 50)
print("CALIBRATION REQUIRED!")
print("=" * 50)
print("Slowly move the sensor in figure-8 patterns")
print("Rotate it in ALL axes (roll, pitch, yaw)")
print("Keep going until accuracy reaches 3/3...")
print("=" * 50 + "\n")

calibrated = False
attempts = 0

while not calibrated and attempts < 300:  # 30 seconds max
    time.sleep_ms(100)
    attempts += 1
    
    try:
        status = bno.calibration_status
        mx, my, mz = bno.mag
        
        if status == 3:
            print(f"\n✓ CALIBRATION COMPLETE! Accuracy: {status}/3")
            print(f"  Magnetometer: X={mx:.2f} Y={my:.2f} Z={mz:.2f}")
            
            # Save to flash!
            bno.calibration_save
            print("✓ Calibration saved to sensor memory")
            calibrated = True
        else:
            # Progress bar
            bar = "█" * status + "░" * (3 - status)
            print(f"  Calibrating... [{bar}] {status}/3  Mag: {mx:.1f},{my:.1f},{mz:.1f}", end="\r")
            
    except Exception as e:
        print(f"Error: {e}")
        continue

if not calibrated:
    print("\n✗ Calibration timeout. Try moving the sensor more vigorously.")
else:
    print("\nNow you can use BNO_REPORT_ROTATION_VECTOR!")