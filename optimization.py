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
    #All features plus image
    num_features = X_num.shape[1]+1
    

    # Initialize population with random binary masks
    population = np.random.randint(0, 2, (population_size, num_features))
    test = torch.tensor(population)
    while torch.all(test == 0): # Recreate if all population are zero
        print("Recreating population")
        population = np.random.randint(0, 2, (population_size, num_features))
        

    for generation in range(num_generations):
        # Evaluate fitness of each individual
        fitness_scores = []
        for individual in population:
            selected_features = individual[:-1].astype(bool) # Selecting all except the last column
            selected_images = individual[-1].astype(bool) # Selecting just the last column
            fitness = fitness_fn(X_num[:, selected_features], X_img, y, selected_images)
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
    # best_individual = population[np.argmax(fitness_scores)] old
    best_fitness_score = 0
    best_sum = 0
    for i, fitness in enumerate(fitness_scores):
        if (fitness > best_fitness_score) or ((fitness == best_fitness_score) and (sum(population[i])>best_sum)):
            best_fitness_score = fitness
            best_sum = sum(population[i])
            best_individual = population[i]
            print(f"Best fitness {best_fitness_score} - {best_individual}")

    return best_individual


def particle_swarm_optimization(X_num, X_img, y, fitness_fn, num_particles=20, num_iterations=50, w=0.5, c1=0.5, c2=1, device="cpu"):
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
    #Adding image feature
    num_features = X_num.shape[1]+1

    # Initialize particles randomly
    particles = np.random.rand(num_particles, num_features) > 0.5
    test = torch.tensor(particles)
    while torch.all(test == 0): # Recreate if all particles are zero
        print("Recreating particles")
        particles = np.random.rand(num_particles, num_features) > 0.5

    velocities = np.random.rand(num_particles, num_features) * 0.1

    # Initialize personal and global bests
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([fitness_fn(X_num[:, p[:-1].astype(bool)], X_img, y, p[-1].astype(bool)) for p in particles])
    personal_best_sum =  np.array([sum(p) for p in particles])

    global_best_position = 0
    global_best_score = 0
    global_best_sum = 0
    for i, personal_best_position in enumerate(personal_best_positions):
        if (personal_best_scores[i] > global_best_score) or ((personal_best_scores[i] == global_best_score) and (sum(particles[i]) > global_best_sum)):
            global_best_position = personal_best_position
            global_best_score = personal_best_scores[i]
            global_best_sum = sum(particles[i])

    for iteration in range(num_iterations):
        for i, particle in enumerate(particles):
            # Evaluate fitness
            selected_features = particle[:-1].astype(bool)
            selected_images = particle[-1].astype(bool)
            # print(f"Testing particle {i} = {particle.astype(bool)}")
            fitness = fitness_fn(X_num[:, selected_features], X_img, y, selected_images)

            # Update personal best
            if (fitness > personal_best_scores[i]) or ((fitness == personal_best_scores[i]) and (sum(particle) > personal_best_sum[i])):
                personal_best_positions[i] = particle
                personal_best_scores[i] = fitness
                personal_best_sum[i] = sum(particle)

            # Update global best
            if (fitness > global_best_score) or ((fitness == global_best_score) and (sum(particle) > global_best_sum)):
                global_best_position = particle
                global_best_score = fitness
                global_best_sum = sum(particle)

        # Update velocities and positions
        for i in range(num_particles):
            r1, r2 = np.random.rand(num_features), np.random.rand(num_features)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i].astype(int) - particles[i].astype(int))
                + c2 * r2 * (global_best_position.astype(int) - particles[i].astype(int))
            )
            
            sigmoid = 1 / (1 + np.exp(-velocities[i]))            
            mutation_mask = (sigmoid > 0.5).astype(int)
            particles[i] = np.bitwise_xor(particles[i].astype(int), (mutation_mask))
            
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