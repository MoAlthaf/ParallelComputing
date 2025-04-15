attempts to go beyond `N = 350,000,000` (e.g., `360,000,000` and higher) resulted in out-of-memory errors or system crashes. Therefore, 350 million was the optimal maximum value for our current setup.

### Step 4.g – Experimentation & Observations
**Scenario: Fixed Vaccination Rate at 70%**

- **Spread chance:** 0.3
- **Vaccination rate:** 0.70 (uniform across processes)
- **Resulting infection rate:** ~73% on all processes
This experiment demonstrates how higher vaccination rates reduce total infections in a simulated population, and how even with moderate virus spread, containment is possible when immunity reaches a threshold.