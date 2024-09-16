Example Acceleration: 

| OPTION TYPE | TIME STEPS | SIMULATIONS | S | K | r | q | iv | t  |
|-------------|------------|-------------|---|---|----|--|---|-----|
| EU CALL     | 100         | 100,0000   | 100.0 | 100.0 | 0.05 | 0.00 | 0.20 | 1 yr | 

Results:

|        | **GPU Simulation Results**    | **CPU Simulation Results**     | **Black Scholes Results**   |
|----------------------------------|--------------------|---------------------|------------------|
| **Hardware Information**      | **NVIDIA T400 4GB**| **Intel i9-13900K** | **N/A**          |
| Computation Time    | 14.2 milliseconds    | 1.7 seconds    | N/A              |
| Option Price  | 10.4722    |  10.3626    | 10.451 |
| Delta | 0.636881          |  0.6365          |  0.637        |
| Gamm | 0.0190003           |  0.1555           | 0.019        |
| Vega |  0.377253            | 0.4384            |  0.375         |
| Theta |  -0.0265514           |  0.3493           |  -0.018	        |
| Tho | 0.531521             |  0.6896             | 0.532          |

The differences in rates of convergence are likely due to RNG inconsistency in CPU
