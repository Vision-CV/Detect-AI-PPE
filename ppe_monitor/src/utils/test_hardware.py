#!/usr/bin/env python3
from hardware import get_hardware_profile

config = get_hardware_profile()
print("\n📊 Generated Config:")
for k, v in config.items():
    if k != "hardware":
        print(f"   {k}: {v}")
