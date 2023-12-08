## Towards a Dynamic Tracking of Assembly Lines

This code represents a data-driven simulator for assembly lines based on hazard delays, breakdowns, and other events.

Author: Anass El Houd - FORVIA Clean Mobility

### Features

- [x] Breakdowns of machines included: A MTTF (s) is to be defined. An average downtime of machines is calculated.
- [x] Buffers have capacities and initial state that can be defined. 
- [x] Idle time of machines is calcualted. It refers to the average waiting time of the machines while they are operating.
- [x] Hazard delays of machine time and manual time are included and calculated based on a Weibull distribution.
- [x] Efficiently handling the supermarket (raw stock) dynamics at the input of the initial machine.
- [ ] Impact of operator fatigue throughout the shift on manual operation time.
- [ ] Manual Learning Model: manual operators tends to perform manual tasks more efficiently over time, that's the learning curve model that will decrease the manual operating time 


The simulator uses data to simulate various hazards and breakdowns that can occur in a system. It provides a realistic representation of these events based on the input data for long-term verifications.

Usage:
- Import the simulator module.
- Initialize the simulator with the required data.
- Call the appropriate methods to simulate hazards and breakdowns.
- Analyze the results and take necessary actions.



