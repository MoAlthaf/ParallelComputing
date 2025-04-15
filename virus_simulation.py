from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


population_size = 1000
spread_chance = 0.1
vaccination_rate = 0.7 #np.random.uniform(0.1, 0.5)  


population = np.zeros(population_size, dtype=int)

if rank == 0:
    infected_indices = np.random.choice(population_size, int(0.1 * population_size), replace=False)
    population[infected_indices] = 1


population = comm.bcast(population, root=0)

def spread_virus(population, spread_chance, vaccination_rate):
    new_population = population.copy()
    for i in range(len(population)):
        if population[i] == 0:  # Uninfected
            if np.random.rand() > vaccination_rate:
                if np.random.rand() < spread_chance:
                    new_population[i] = 1
    return new_population


for _ in range(10):
    population = spread_virus(population, spread_chance, vaccination_rate)

    if rank != 0:
        comm.send(population, dest=0)
    else:
        for i in range(1, size):
            received = comm.recv(source=i)
            population = np.logical_or(population, received).astype(int)

   
    population = comm.bcast(population, root=0)

local_infected = np.sum(population)
infection_rate = local_infected / population_size
print(f"Process {rank} | Vaccination Rate: {vaccination_rate:.2f} | Infection Rate: {infection_rate:.2f}")
