# Field incident report INC-2024-09-18: Baltika Heat Networks

Customer: Baltika Heat Networks, Riga, Latvia (Standard support tier). Installed base: 6 Kestrel K400 high-temperature sensors, part number KS-400-H, on DN200 district heating mains, running firmware 2.9 since 2023.

Reported problem: after the local distributor attempted an upgrade directly from firmware 2.9 to 3.1, two units became unresponsive and showed only a bootloader message in the Meridian Service Tool. The remaining four units also reported weakening ultrasonic signal strength over the summer.

Root cause: a direct jump from firmware 2.x to 3.1 is not supported; firmware 3.0 introduced a new bootloader and must be installed first. The weak signal was traced to standard CG-5 coupling gel drying out at supply temperatures around 110 degrees C; high-temperature installations require the CG-5H variant.

Resolution: the two unresponsive units were recovered through the Service Tool recovery mode and all six were staged through 2.9 to 3.0 to 3.1 on 2024-09-25. The coupling gel was replaced with CG-5H on all units and signal strength returned to normal.

Recommendation: distributors must stage firmware upgrades for any fleet still on 2.x, and every high-temperature installation should switch to CG-5H at its next calibration visit.
