# Kestrel firmware release notes

## Firmware 3.2 (released 2025-02-14)

Adds Modbus TCP support through the ETH-1 module, a diagnostics page in the Meridian Service Tool showing signal strength per transducer, and a configurable damping filter. Upgrading directly from 3.0 or 3.1 is supported.

## Firmware 3.1 (released 2024-11-19)

Maintenance release. Fixes ticket MI-1187: on pipes below 50 mm, units running firmware 3.0 could report a growing zero-flow offset after about 72 hours of continuous operation. The fix resets the transit-time baseline every 6 hours. Also fixes a Modbus RTU exception code on register 40012.

## Firmware 3.0 (released 2024-06-03)

New signal processing pipeline with improved handling of entrained air, and a new bootloader. Units on firmware 2.x must upgrade to 3.0 before any later version; a direct jump from 2.x to 3.1 or 3.2 is not supported.
