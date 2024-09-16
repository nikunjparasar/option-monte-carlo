#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

// CUDA kernel to simulate each path and calculate Greeks
__global__ void monte_carlo_kernel(float *d_payoffs, float *d_deltas, float *d_gammas, float *d_vegas, float *d_thetas, float *d_rhos,
                                   float S, float K, float r, float q, float iv, float dt, float t, int N, int simulations,
                                   float delta_S, float delta_iv, float delta_r, float delta_t)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= simulations)
        return;

    // Initialize random number generator for each thread
    curandState state;
    curand_init(1234, idx, 0, &state);

    // Initialize asset prices for base case and finite difference cases
    float S_t = S;
    float S_up = S + delta_S;
    float S_down = S - delta_S;
    float S_vega = S;
    float S_theta = S;
    float S_rho = S;

    // Calculate adjusted time for Theta
    float t_theta = t - delta_t;
    if (t_theta <= 0.0f)
        t_theta = 1e-6f;
    float dt_theta = t_theta / N;

    for (int i = 0; i < N; i++)
    {
        float dWT = curand_normal(&state) * sqrtf(dt); // Gaussian random number for the Brownian motion

        // Simulate base case
        S_t += (r - q) * S_t * dt + iv * S_t * dWT;

        // Simulate for Delta and Gamma
        S_up += (r - q) * S_up * dt + iv * S_up * dWT;
        S_down += (r - q) * S_down * dt + iv * S_down * dWT;

        // Simulate for Vega (with increased volatility)
        S_vega += (r - q) * S_vega * dt + (iv + delta_iv) * S_vega * dWT;

        // Simulate for Theta (with reduced time to expiration)
        S_theta += (r - q) * S_theta * dt_theta + iv * S_theta * dWT;

        // Simulate for Rho (with increased interest rate)
        S_rho += (r + delta_r - q) * S_rho * dt + iv * S_rho * dWT;
    }

    // Payoff calculation for the base case
    float C = expf(-r * t) * fmaxf(S_t - K, 0.0f);
    d_payoffs[idx] = C;

    // Delta: (C(S + ΔS) - C(S - ΔS)) / (2 * ΔS)
    float C_up = expf(-r * t) * fmaxf(S_up - K, 0.0f);
    float C_down = expf(-r * t) * fmaxf(S_down - K, 0.0f);
    d_deltas[idx] = (C_up - C_down) / (2.0f * delta_S);

    // Gamma: (C(S + ΔS) - 2 * C(S) + C(S - ΔS)) / (ΔS^2)
    d_gammas[idx] = (C_up - 2.0f * C + C_down) / (delta_S * delta_S);

    // Vega: (C(σ + Δσ) - C(σ)) / Δσ
    float C_vega = expf(-r * t) * fmaxf(S_vega - K, 0.0f);
    d_vegas[idx] = (C_vega - C) / delta_iv;

    // Theta: (C(t - Δt) - C(t)) / Δt
    float C_theta = expf(-r * t_theta) * fmaxf(S_theta - K, 0.0f);
    d_thetas[idx] = (C_theta - C) / delta_t;

    // Rho: (C(r + Δr) - C(r)) / Δr
    float C_rho = expf(-(r + delta_r) * t) * fmaxf(S_rho - K, 0.0f);
    d_rhos[idx] = (C_rho - C) / delta_r;
}

// Function to compute Monte Carlo option price and Greeks on the GPU
void monte_carlo_option_price_and_greeks(float S, float K, float r, float q, float iv, float t, int N, int simulations)
{
    float dt = t / N;
    float delta_S = 0.01f * S;   // Small change in stock price for Delta and Gamma
    float delta_iv = 0.01f * iv; // Small change in volatility for Vega
    float delta_r = 0.01f * r;   // Small change in interest rate for Rho
    float delta_t = t / 365.0f;  // Small change in time for Theta

    // Allocate memory for payoffs and Greeks
    float *d_payoffs, *d_deltas, *d_gammas, *d_vegas, *d_thetas, *d_rhos;
    cudaMalloc((void **)&d_payoffs, simulations * sizeof(float));
    cudaMalloc((void **)&d_deltas, simulations * sizeof(float));
    cudaMalloc((void **)&d_gammas, simulations * sizeof(float));
    cudaMalloc((void **)&d_vegas, simulations * sizeof(float));
    cudaMalloc((void **)&d_thetas, simulations * sizeof(float));
    cudaMalloc((void **)&d_rhos, simulations * sizeof(float));

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the Monte Carlo kernel with grid and block dimensions
    int blocks = (simulations + BLOCK_SIZE - 1) / BLOCK_SIZE;
    monte_carlo_kernel<<<blocks, BLOCK_SIZE>>>(d_payoffs, d_deltas, d_gammas, d_vegas, d_thetas, d_rhos,
                                               S, K, r, q, iv, dt, t, N, simulations, delta_S, delta_iv, delta_r, delta_t);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the results back to the host
    float *h_payoffs = (float *)malloc(simulations * sizeof(float));
    float *h_deltas = (float *)malloc(simulations * sizeof(float));
    float *h_gammas = (float *)malloc(simulations * sizeof(float));
    float *h_vegas = (float *)malloc(simulations * sizeof(float));
    float *h_thetas = (float *)malloc(simulations * sizeof(float));
    float *h_rhos = (float *)malloc(simulations * sizeof(float));

    cudaMemcpy(h_payoffs, d_payoffs, simulations * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_deltas, d_deltas, simulations * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gammas, d_gammas, simulations * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vegas, d_vegas, simulations * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_thetas, d_thetas, simulations * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rhos, d_rhos, simulations * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate the averages (option price and Greeks)
    double payoff_sum = 0.0, delta_sum = 0.0, gamma_sum = 0.0, vega_sum = 0.0, theta_sum = 0.0, rho_sum = 0.0;
    for (int i = 0; i < simulations; i++)
    {
        payoff_sum += h_payoffs[i];
        delta_sum += h_deltas[i];
        gamma_sum += h_gammas[i];
        vega_sum += h_vegas[i];
        theta_sum += h_thetas[i];
        rho_sum += h_rhos[i];
    }

    float option_price = payoff_sum / simulations;
    float delta = delta_sum / simulations;
    float gamma = gamma_sum / simulations;
    float vega = (vega_sum / simulations) * 0.01;
    float theta = (theta_sum / simulations) * 0.01;
    float rho = (rho_sum / simulations) * 0.01;

    // Print the results
    std::cout << "Option Price: " << option_price << std::endl;
    std::cout << "Delta: " << delta << std::endl;
    std::cout << "Gamma: " << gamma << std::endl;
    std::cout << "Vega: " << vega << std::endl;
    std::cout << "Theta: " << theta << std::endl;
    std::cout << "Rho: " << rho << std::endl;
    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

    // Print GPU device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;

    // Free memory
    free(h_payoffs);
    free(h_deltas);
    free(h_gammas);
    free(h_vegas);
    free(h_thetas);
    free(h_rhos);
    cudaFree(d_payoffs);
    cudaFree(d_deltas);
    cudaFree(d_gammas);
    cudaFree(d_vegas);
    cudaFree(d_thetas);
    cudaFree(d_rhos);
}

int main()
{
    // Initialize option parameters
    float S = 100.0f; // Initial stock price
    float K = 100.0f; // Strike price
    float r = 0.05f;  // Risk-free interest rate
    float q = 0.0f;   // Dividend yield
    float iv = 0.2f;  // Implied volatility
    float t = 1.0f;   // Time to expiration in years

    int N = 100;              // Number of time steps
    int simulations = 100000; // Number of Monte Carlo simulations

    // Call the Monte Carlo pricing function
    monte_carlo_option_price_and_greeks(S, K, r, q, iv, t, N, simulations);

    return 0;
}
