import time
from machine import Pin, I2C
from bno08x import *

print("=" * 50)
print("BNO085 DIAGNOSTIC TOOL")
print("=" * 50)

# Setup I2C
i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=100000)

# Test 1: I2C Scan
print("\n[TEST 1] I2C Bus Scan...")
devices = i2c.scan()
print(f"Devices found: {[hex(d) for d in devices]}")
if 0x4A in devices:
    print("✓ BNO085 found at 0x4A")
    addr = 0x4A
elif 0x4B in devices:
    print("✓ BNO085 found at 0x4B")
    addr = 0x4B
else:
    print("✗ BNO085 NOT FOUND! Check wiring.")
    addr = 0x4A  # Try anyway

# Test 2: Basic initialization
print("\n[TEST 2] Basic Initialization...")
try:
    bno = BNO08X(i2c, address=addr)
    print("✓ BNO08X object created")
except Exception as e:
    print(f"✗ Failed to create BNO08X object: {e}")

# Test 3: Enable reports one by one
print("\n[TEST 3] Testing Individual Reports...")

reports_to_test = [
    ("BNO_REPORT_ACCELEROMETER", BNO_REPORT_ACCELEROMETER),
    ("BNO_REPORT_GYROSCOPE", BNO_REPORT_GYROSCOPE),
    ("BNO_REPORT_MAGNETOMETER", BNO_REPORT_MAGNETOMETER),
    ("BNO_REPORT_GAME_ROTATION_VECTOR", BNO_REPORT_GAME_ROTATION_VECTOR),
    ("BNO_REPORT_ROTATION_VECTOR", BNO_REPORT_ROTATION_VECTOR),
]

working_reports = []

for name, report in reports_to_test:
    try:
        bno.enable_feature(report)
        time.sleep_ms(200)
        print(f"✓ {name} - ENABLED")
        working_reports.append(name)
    except Exception as e:
        print(f"✗ {name} - FAILED: {e}")

# Test 4: Try reading from working reports
print("\n[TEST 4] Reading Data...")

if "BNO_REPORT_GAME_ROTATION_VECTOR" in working_reports:
    try:
        roll, pitch, yaw = bno.euler
        print(f"✓ Game Rotation Euler: R={roll:.1f} P={pitch:.1f} Y={yaw:.1f}")
    except Exception as e:
        print(f"✗ Game Rotation read failed: {e}")

if "BNO_REPORT_ROTATION_VECTOR" in working_reports:
    try:
        roll, pitch, yaw = bno.euler
        print(f"✓ Rotation Vector Euler: R={roll:.1f} P={pitch:.1f} Y={yaw:.1f}")
    except Exception as e:
        print(f"✗ Rotation Vector read failed: {e}")

if "BNO_REPORT_MAGNETOMETER" in working_reports:
    try:
        mx, my, mz = bno.mag
        print(f"✓ Magnetometer: X={mx:.2f} Y={my:.2f} Z={mz:.2f}")
        if mx == 0.0 and my == 0.0 and mz == 0.0:
            print("  ⚠ WARNING: All zeros - magnetometer not working!")
    except Exception as e:
        print(f"✗ Magnetometer read failed: {e}")

# Test 5: Check calibration
print("\n[TEST 5] Calibration Status...")
try:
    # Try enabling magnetometer first if not already
    if "BNO_REPORT_MAGNETOMETER" not in working_reports:
        bno.enable_feature(BNO_REPORT_MAGNETOMETER)
        time.sleep_ms(200)
    
    cal_status = bno.calibration_status
    print(f"✓ Magnetometer calibration status: {cal_status}/3")
    if cal_status == 0:
        print("  ⚠ Not calibrated - this may cause ROTATION_VECTOR to fail!")
except Exception as e:
    print(f"✗ Could not get calibration status: {e}")

# Summary
print("\n" + "=" * 50)
print("DIAGNOSTIC SUMMARY")
print("=" * 50)
print(f"Working reports: {working_reports}")

if "BNO_REPORT_ROTATION_VECTOR" not in working_reports:
    print("\n⚠ ROTATION_VECTOR failed to enable!")
    print("  Possible causes:")
    print("  1. Magnetometer not responding (check for metal interference)")
    print("  2. Calibration required first")
    print("  3. Library bug - try updating")
    print("\n  RECOMMENDATION: Use BNO_REPORT_GAME_ROTATION_VECTOR instead")
    print("  (Yaw will drift but no magnetometer issues)")