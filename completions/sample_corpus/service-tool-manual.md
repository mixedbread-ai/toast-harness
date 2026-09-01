# Meridian Service Tool manual (excerpt)

The Meridian Service Tool is a Windows application that connects to Kestrel sensors and Osprey loggers over the USB service port. It handles firmware upgrades, configuration, diagnostics and log export for the whole product range.

Firmware upgrade procedure: download the firmware package from the distributor portal, let the tool verify the package checksum, and follow the staged path the tool proposes. For Kestrel units still on firmware 2.x the tool always stages through 3.0 before any later version, because 3.0 replaces the bootloader.

Recovery mode: if an upgrade is interrupted and the unit shows only a bootloader message, hold the service button while connecting the cable. The tool detects the bootloader, offers recovery, and reflashes the last verified firmware package.

Diagnostics: on Kestrel firmware 3.2 or later the tool shows a diagnostics page with per-transducer signal strength and electronics temperature. Export the diagnostics log before opening a support case; the support team reads these files directly.

ETH-1 configuration: when the Ethernet module (ET-001) is fitted, the tool's network page assigns a static IP address or enables DHCP, sets the Modbus TCP port (502 by default), and runs a connection test against the configured gateway.
