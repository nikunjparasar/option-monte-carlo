#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

enum OptionType { CALL = 0, PUT = 1 };

// Option Parameters
struct Option {
    double S;        // Initial stock price
    double K;        // Strike price
    double T;        // Time to maturity
    double r;        // Risk-free rate
    double sigma;    // Volatility
    enum OptionType type; // Option type: CALL or PUT
};

// Greeks Structure
struct Greeks {
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

// Generate a normally distributed random number using Box-Muller transform
double rand_normal() {
    double u = (double)rand() / RAND_MAX;
    double v = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}

// Perform Monte Carlo simulation
double monte_carlo_simulation(const struct Option *option, int steps) {
    double dt = option->T / steps;
    double drift = (option->r - 0.5 * option->sigma * option->sigma) * dt;
    double diffusion = option->sigma * sqrt(dt);
    double logS = log(option->S);

    printf("Simulating price path...\n");

    // Simulate the stock price path
    for (int i = 0; i < steps; ++i) {
        double Z = rand_normal(); // Generate a normal random variable
        logS += drift + diffusion * Z;
    }

    double ST = exp(logS); // Final stock price
    double payoff = 0.0;
    if (option->type == CALL) {
        payoff = fmax(ST - option->K, 0.0);
    } else {
        payoff = fmax(option->K - ST, 0.0);
    }

    printf("Final stock price: %.6lf, Payoff: %.6lf\n", ST, payoff);

    return exp(-option->r * option->T) * payoff; // Discounted payoff
}

// Perform Monte Carlo simulation for option pricing and Greeks calculation
void monte_carlo_option_cpu(const struct Option *option, int N, int steps, struct Greeks *greeks) {
    double sum_price = 0.0;
    double sum_price_up = 0.0;
    double sum_price_down = 0.0;
    double sum_price_vega = 0.0;
    double sum_price_theta = 0.0;
    double sum_price_rho = 0.0;

    printf("Starting Monte Carlo simulation with %d paths and %d steps per path...\n", N, steps);

    // Original simulation
    for (int i = 0; i < N; ++i) {
        if (i % (N / 10) == 0) { // Print progress every 10%
            printf("Simulating path %d of %d...\n", i + 1, N);
        }
        sum_price += monte_carlo_simulation(option, steps);
    }
    double option_price = sum_price / N;

    printf("Finished main simulation, calculating Greeks...\n");

    // Delta: Simulate for S + epsilon
    double epsilon = 1e-4;
    struct Option opt_up = *option;
    opt_up.S += epsilon;
    for (int i = 0; i < N; ++i) {
        sum_price_up += monte_carlo_simulation(&opt_up, steps);
    }
    double price_up = sum_price_up / N;

    // Gamma: Simulate for S - epsilon
    struct Option opt_down = *option;
    opt_down.S -= epsilon;
    for (int i = 0; i < N; ++i) {
        sum_price_down += monte_carlo_simulation(&opt_down, steps);
    }
    double price_down = sum_price_down / N;

    // Vega: Simulate for sigma + epsilon
    struct Option opt_vega = *option;
    opt_vega.sigma += epsilon;
    for (int i = 0; i < N; ++i) {
        sum_price_vega += monte_carlo_simulation(&opt_vega, steps);
    }
    double price_vega = sum_price_vega / N;

    // Theta: Simulate for T + epsilon
    struct Option opt_theta = *option;
    opt_theta.T += epsilon;
    for (int i = 0; i < N; ++i) {
        sum_price_theta += monte_carlo_simulation(&opt_theta, steps);
    }
    double price_theta = sum_price_theta / N;

    // Rho: Simulate for r + epsilon
    struct Option opt_rho = *option;
    opt_rho.r += epsilon;
    for (int i = 0; i < N; ++i) {
        sum_price_rho += monte_carlo_simulation(&opt_rho, steps);
    }
    double price_rho = sum_price_rho / N;

    // Calculate Greeks
    greeks->delta = (price_up - option_price) / epsilon;
    greeks->gamma = (price_up - 2.0 * option_price + price_down) / (epsilon * epsilon);
    greeks->vega = (price_vega - option_price) / epsilon;
    greeks->theta = (price_theta - option_price) / epsilon;
    greeks->rho = (price_rho - option_price) / epsilon;

    // Output the results
    printf("Option Price: %.6lf\n", option_price);
    printf("Greeks:\n");
    printf("Delta: %.6lf\n", greeks->delta);
    printf("Gamma: %.6lf\n", greeks->gamma);
    printf("Vega: %.6lf\n", greeks->vega);
    printf("Theta: %.6lf\n", greeks->theta);
    printf("Rho: %.6lf\n", greeks->rho);
}

int main() {
    srand(time(NULL)); // Seed the random number generator

    // Define option parameters
    struct Option option;
    option.S = 100.0;      // Initial stock price
    option.K = 100.0;      // Strike price
    option.T = 1.0;        // Time to maturity (1 year)
    option.r = 0.05;       // Risk-free rate (5%)
    option.sigma = 0.2;    // Volatility (20%)
    option.type = CALL;    // Option type: CALL or PUT

    // Monte Carlo parameters
    int N = 1000000;        // Number of simulation paths
    int steps = 100;        // Number of time steps per path

    // Greeks
    struct Greeks greeks;

    // Run Monte Carlo simulation
    monte_carlo_option_cpu(&option, N, steps, &greeks);

    return 0;
}
