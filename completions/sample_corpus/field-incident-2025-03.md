# Field incident report INC-2025-03-07: Nordhavn Water Utility

Customer: Nordhavn Water Utility, Denmark (Gold support tier). Installed base: 14 Kestrel K400 sensors, part number KS-400-A, on 40 mm distribution pipes, all running firmware 3.0 since commissioning in January 2025.

Reported problem: from 2025-03-04 the utility observed a slowly growing flow reading at night, when the actual flow is near zero. The offset reached 0.12 m/s after three days and reset after a power cycle. Two units also showed cracked transducer housings after a frost period.

Root cause: the drift matches ticket MI-1187, the zero-flow offset on small pipes fixed in firmware 3.1. The cracked housings are a materials defect covered by warranty.

Resolution: all 14 units were upgraded to firmware 3.1 on 2025-03-12 through the Meridian Service Tool; the drift did not recur in the following two weeks. The two units with cracked housings were replaced under warranty through the Gold-tier advance replacement process.

Recommendation: proactively check every K400 fleet still on firmware 3.0 on pipes below 50 mm and schedule the upgrade to 3.1 or later.
