import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def genetic_algorithm(X_num, X_img, y, fitness_fn, num_generations=50, population_size=20, mutation_rate=0.1, device="cpu"):
    """
    Genetic Algorithm for feature selection on multimodal data.

    Args:
        X_num (torch.Tensor): Structured numerical data.
        X_img (torch.Tensor): Image data.
        y (torch.Tensor): Labels.
        fitness_fn (callable): Fitness function to evaluate feature subsets.
        num_generations (int): Number of generations.
        population_size (int): Size of the population.
        mutation_rate (float): Probability of mutation.
        device (str): Device to run the computation.

    Returns:
        np.ndarray: Binary mask of selected features.
    """
    num_features = X_num.shape[1]

    # Initialize population with random binary masks
    population = np.random.randint(0, 2, (population_size, num_features))

    for generation in range(num_generations):
        # Evaluate fitness of each individual
        fitness_scores = []
        for individual in population:
            selected_features = individual.astype(bool)
            fitness = fitness_fn(X_num[:, selected_features], X_img, y)
            fitness_scores.append(fitness)

        fitness_scores = np.array(fitness_scores)
        print(f"Generation {generation + 1}: Best Fitness = {fitness_scores.max()}")

        # Selection: Retain top individuals based on fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = population[sorted_indices[:population_size // 2]]

        # Crossover: Combine pairs of top individuals to create offspring
        offspring = []
        for _ in range(population_size - len(population)):
            parent1, parent2 = population[np.random.choice(len(population), 2, replace=False)]
            crossover_point = np.random.randint(1, num_features - 1)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring.append(child)
        offspring = np.array(offspring)

        # Mutation: Randomly flip bits in the offspring
        mutations = np.random.rand(*offspring.shape) < mutation_rate
        offspring = np.logical_xor(offspring, mutations).astype(int)

        # Create the new population
        population = np.vstack((population, offspring))

    # Return the best individual from the final generation
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual


import numpy as np

def particle_swarm_optimization(X_num, X_img, y, fitness_fn, num_particles=30, num_iterations=50, w=0.5, c1=2, c2=2, device="cpu"):
    """
    Particle Swarm Optimization for feature selection on multimodal data.

    Args:
        X_num (torch.Tensor): Structured numerical data.
        X_img (torch.Tensor): Image data.
        y (torch.Tensor): Labels.
        fitness_fn (callable): Fitness function to evaluate feature subsets.
        num_particles (int): Number of particles in the swarm.
        num_iterations (int): Number of iterations.
        w (float): Inertia weight.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.
        device (str): Device to run the computation.

    Returns:
        np.ndarray: Binary mask of selected features.
    """
    num_features = X_num.shape[1]

    # Initialize particles randomly
    particles = np.random.rand(num_particles, num_features) > 0.5
    velocities = np.random.rand(num_particles, num_features) * 0.1

    # Initialize personal and global bests
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([fitness_fn(X_num[:, p.astype(bool)], X_img, y) for p in particles])
    global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
    global_best_score = personal_best_scores.max()

    for iteration in range(num_iterations):
        for i, particle in enumerate(particles):
            # Evaluate fitness
            selected_features = particle.astype(bool)
            fitness = fitness_fn(X_num[:, selected_features], X_img, y)

            # Update personal best
            if fitness > personal_best_scores[i]:
                personal_best_positions[i] = particle
                personal_best_scores[i] = fitness

            # Update global best
            if fitness > global_best_score:
                global_best_position = particle
                global_best_score = fitness

        # Update velocities and positions
        for i in range(num_particles):
            r1, r2 = np.random.rand(num_features), np.random.rand(num_features)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i] - particles[i])
                + c2 * r2 * (global_best_position - particles[i])
            )
            particles[i] = (np.random.rand(num_features) < (1 / (1 + np.exp(-velocities[i])))).astype(int)

        print(f"Iteration {iteration + 1}: Best Fitness = {global_best_score}")

    return global_best_position



def pca_feature_selection(X_num, n_components):
    """
    Perform PCA for feature selection.

    Args:
        X_num (np.ndarray): Numerical features.
        n_components (int): Number of principal components to keep.

    Returns:
        torch.Tensor: Transformed features.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_num.cpu().numpy())
    return torch.tensor(X_pca, dtype=torch.float32), pca