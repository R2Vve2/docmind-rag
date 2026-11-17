# NX-200 Collaborative Robot Arm: Product Manual

## Overview

The NX-200 is a six-axis collaborative robot arm designed for light assembly, pick-and-place and inspection tasks. It is rated for a maximum payload of 12 kg and has a reach of 1,300 mm. Repeatability is plus or minus 0.03 mm.

## Technical specifications

- Degrees of freedom: 6
- Maximum payload: 12 kg
- Reach: 1,300 mm
- Repeatability: +/- 0.03 mm
- Weight of the arm: 38 kg
- Power supply: 100-240 V AC, 50/60 Hz, maximum 600 W
- Joint speed: up to 180 degrees per second on joints 1-3, 225 degrees per second on joints 4-6
- Operating temperature: 0 to 45 degrees Celsius
- IP rating: IP54 for the arm, IP20 for the controller
- Controller: NXC-4 with 32 digital inputs and 32 digital outputs
- Communication: EtherCAT, Modbus TCP, PROFINET and a REST API over Ethernet

## Safety

The NX-200 complies with ISO 10218-1 and ISO/TS 15066 for collaborative operation. Collision detection stops the arm within 50 ms when a contact force above the configured threshold is measured. The default threshold is 120 N and can be lowered to 60 N in the safety configuration.

The emergency stop button on the teach pendant cuts power to all joint motors and engages the brakes. After an emergency stop, the arm must be re-enabled from the pendant and the brakes released before motion can resume.

## Installation

Mount the base on a surface that can withstand a static load of at least 2,000 N and a dynamic torque of 400 Nm. Use the four M10 bolts supplied and tighten to 45 Nm. Allow at least 200 mm of clearance around the controller for ventilation.

## Maintenance

Perform a visual inspection of cables and joint covers every month. Joint lubrication is required every 5,000 operating hours or every 12 months, whichever comes first. Firmware updates are distributed through the Northwind Portal and should be applied during scheduled downtime. Always back up the robot program before updating firmware.

## Warranty

The NX-200 is covered by a 24-month standard warranty from the date of delivery. An extended warranty of up to 60 months is available through Northwind Care. The warranty does not cover damage caused by exceeding the rated payload or by operating outside the specified temperature range.

## Support

Technical support is available Sunday to Thursday, 09:00 to 18:00 Cairo time, by email at support@northwind-robotics.example or through the Northwind Portal. Priority support with a four-hour response time is included in Northwind Care.
