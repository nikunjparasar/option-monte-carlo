#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Function to generate a random number from a standard normal distribution (Box-Muller Transform)
float generate_random_normal()
{
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrt(-2.0f * log(u1)) * cos(2.0f * 2.0f * M_PI * u2);
}

// Function to simulate a single path of the asset price
float simulate_path(float S, float r, float q, float iv, float dt, int N)
{
    for (int i = 0; i < N; ++i)
    {
        float dWT = generate_random_normal() * sqrt(dt);
        S += (r - q) * S * dt + iv * S * dWT;
    }
    return S;
}

// Function to compute Monte Carlo option price and Greeks on the CPU
void monte_carlo_option_price_and_greeks(float S, float K, float r, float q, float iv, float t, int N, int simulations)
{
    float dt = t / N;
    float delta_S = 0.01f * S;   // Small change in stock price for Delta and Gamma
    float delta_iv = 0.01f * iv; // Small change in volatility for Vega
    float delta_r = 0.01f * r;   // Small change in interest rate for Rho
    float delta_t = t / 365.0f;  // Small change in time for Theta

    double payoff_sum = 0.0, delta_sum = 0.0, gamma_sum = 0.0, vega_sum = 0.0, theta_sum = 0.0, rho_sum = 0.0;

    // Seed the random number generator
    srand(time(NULL));

    for (int i = 0; i < simulations; ++i)
    {
        // Simulate base case and Greeks paths
        float S_t = simulate_path(S, r, q, iv, dt, N);
        float S_up = simulate_path(S + delta_S, r, q, iv, dt, N);
        float S_down = simulate_path(S - delta_S, r, q, iv, dt, N);
        float S_vega = simulate_path(S, r, q, iv + delta_iv, dt, N);

        float t_theta = t - delta_t;
        if (t_theta <= 0.0f)
            t_theta = 1e-6f;
        float dt_theta = t_theta / N;
        float S_theta = simulate_path(S, r, q, iv, dt_theta, N);
        float S_rho = simulate_path(S, r + delta_r, q, iv, dt, N);

        // Calculate payoffs
        float C = exp(-r * t) * fmaxf(S_t - K, 0.0f);
        float C_up = exp(-r * t) * fmaxf(S_up - K, 0.0f);
        float C_down = exp(-r * t) * fmaxf(S_down - K, 0.0f);
        float C_vega = exp(-r * t) * fmaxf(S_vega - K, 0.0f);
        float C_theta = exp(-r * t_theta) * fmaxf(S_theta - K, 0.0f);
        float C_rho = exp(-(r + delta_r) * t) * fmaxf(S_rho - K, 0.0f);

        // Sum the payoffs for option price and Greeks
        payoff_sum += C;
        delta_sum += (C_up - C_down) / (2.0f * delta_S);
        gamma_sum += (C_up - 2.0f * C + C_down) / (delta_S * delta_S);
        vega_sum += (C_vega - C) / delta_iv;
        theta_sum += (C_theta - C) / delta_t;
        rho_sum += (C_rho - C) / delta_r;
    }

    // Calculate averages
    float option_price = payoff_sum / simulations;
    float delta = delta_sum / simulations;
    float gamma = gamma_sum / simulations;
    float vega = (vega_sum / simulations) * 0.01f;
    float theta = (theta_sum / simulations) * 0.01f;
    float rho = (rho_sum / simulations) * 0.01f;

    // Print the results
    printf("Option Price: %.4f\n", option_price);
    printf("Delta: %.4f\n", delta);
    printf("Gamma: %.4f\n", gamma);
    printf("Vega: %.4f\n", vega);
    printf("Theta: %.4f\n", theta);
    printf("Rho: %.4f\n", rho);
}

// Function to get CPU information
void print_cpu_info()
{
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    printf("Number of processors: %u\n", sysinfo.dwNumberOfProcessors);
#elif __linux__
    char buffer[128];
    FILE *file = fopen("/proc/cpuinfo", "r");
    if (file != NULL)
    {
        while (fgets(buffer, sizeof(buffer), file) != NULL)
        {
            if (strstr(buffer, "model name") != NULL)
            {
                printf("%s", buffer);
                break;
            }
        }
        fclose(file);
    }
#elif __APPLE__
    // macOS-specific command for CPU information
    char buffer[128];
    FILE *file = popen("sysctl -n machdep.cpu.brand_string", "r");
    if (file != NULL)
    {
        while (fgets(buffer, sizeof(buffer), file) != NULL)
        {
            printf("%s", buffer);
        }
        pclose(file);
    }
#endif
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

    // Print CPU info
    printf("CPU Information:\n");
    print_cpu_info();

    // Start timer
    clock_t start = clock();

    // Call the Monte Carlo pricing function
    monte_carlo_option_price_and_greeks(S, K, r, q, iv, t, N, simulations);

    // End timer
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print the time taken
    printf("Time taken: %.4f seconds\n", time_taken);

    return 0;
}
