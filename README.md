EU CALL 100 TIME STEPS 100,000 SIMULATIONS

|       | **Value**  | **GPU Simulation Results**    | **CPU Simulation Results**     | **Black Scholes Results**   |
|----------------------------|------------|--------------------|---------------------|------------------|
| **Hardware Information**    |            | **NVIDIA T400 4GB**| **Intel i9-13900K** | **N/A**          |
| Initial Stock Price (S)     | 100.0      | Option Price: 10.4722    | Option Price: 10.3626    | Option Price: 10.451 |
| Strike Price (K)            | 100.0      | Delta: 0.636881          | Delta: 0.6365          | Delta: 0.637        |
| Risk-free Interest Rate (r) | 0.05       | Gamma: 0.0190003           | Gamma: 0.1555           | Gamma: 0.019        |
| Dividend Yield (q)          | 0.00       | Vega: 0.377253            | Vega: 0.4384            | Vega: 0.375         |
| Implied Volatility (iv)     | 0.2        | Theta: -0.0265514           | Theta: 0.3493           | Theta: -0.018	        |
| Time to Expiration (t)      | 1.0 year   | Rho: 0.531521             | Rho: 0.6896             | Rho: 0.532          |
| Time Taken    |     N/S     | Time: 14.2 milliseconds    | Time: 1.7 seconds    | N/A              |
