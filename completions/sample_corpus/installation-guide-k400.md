# Kestrel K400 installation guide (excerpt)

Site selection: choose a straight pipe run with at least 10 pipe diameters upstream and 5 downstream of the sensor, away from pumps, valves and bends. The pipe must run full; avoid high points where air can collect, because entrained air is the most common cause of unstable readings.

Mounting: on pipes below 100 mm use the V-method with both transducers on the same side; on larger pipes use the Z-method with transducers on opposite sides. Fix the transducers with the MK-40 strap kit (or MK-41 chains above 200 mm), apply a bead of CG-5 coupling gel to each transducer face, and tighten until the gel spreads to the edge of the face.

Wiring: supply 24 V DC to terminals 1 and 2, and connect Modbus RTU on the RS-485 A, B and ground terminals using shielded twisted-pair cable. Terminate the bus with a 120 ohm resistor at the last device. If the ETH-1 module is fitted, connect Ethernet after the firmware requirement in the datasheet is met.

Commissioning: connect the Meridian Service Tool through the service port, confirm signal strength above 60 percent on both transducers, and perform a zero-flow check against a closed valve. Set the damping filter only after the zero-flow check passes.

Troubleshooting: if the zero-flow reading is not stable, reseat the transducers with fresh coupling gel and re-check the mounting distance before suspecting the electronics. A signal strength below 40 percent usually means dried gel, paint on the pipe, or an air pocket.
