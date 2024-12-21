import time
import torch
from generate_dataset import generate_multimodal_dataset
from cnn_fitness import evaluate_features
from optimization import genetic_algorithm, particle_swarm_optimization, pca_feature_selection
from device_detection import get_available_device

# def evaluate_selected_features(X_num, X_img, y, selected_features, fitness_fn, device):
#     """
#     Evaluate fitness on the selected features and log the performance.
#     """
#     if isinstance(selected_features, torch.Tensor):
#         X_selected = X_num[:, selected_features]
#     else:
#         X_selected = selected_features

#     # Evaluate fitness
#     fitness_score = fitness_fn(X_selected, X_img, y)
#     print(f"Fitness score: {fitness_score:.4f}")
#     return fitness_score

def main():
    # Detect device
    device = get_available_device()
    print(f"Using device: {device}")
    
    # Generate dataset
    print('Generating multimodal dataset')
    X_num, X_img, y = generate_multimodal_dataset(num_samples=1000, num_features=10, image_size=(64,64), num_classes=5)
    X_num, X_img, y = X_num.to(device), X_img.to(device), y.to(device)

    # Fitness Function
    fitness_fn = lambda X_selected, X_img, y: evaluate_features(X_selected, X_img, y, device)

    # Baseline Evaluation (All Features)
    print("\nEvaluating all features (baseline)...")
    start_time = time.time()
    baseline_fitness = fitness_fn(X_num, X_img, y)
    baseline_time = time.time() - start_time

    # PCA Feature Selection
    print("\n Evaluating PCA-selected features...")
    start_time = time.time()
    X_pca, _ = pca_feature_selection(X_num, n_components=5)
    X_pca = X_pca.to(device)
    pca_time = time.time() - start_time
    pca_fitness = fitness_fn(X_pca, X_img, y)

    # GA Feature Selection
    print("\n Evaluating GA-selected features...")
    start_time = time.time()
    ga_selected_features = genetic_algorithm(X_num, X_img, y, fitness_fn, device=device)
    ga_time = time.time() - start_time
    ga_fitness = fitness_fn(X_num[:, ga_selected_features.astype(bool)], X_img, y)

    # PSO Feature Selection
    print("\n Evaluating PSO-selected features...")
    start_time = time.time()
    pso_selected_features = particle_swarm_optimization(X_num, X_img, y, fitness_fn, device=device)
    pso_time = time.time() - start_time
    pso_fitness = fitness_fn(X_num[:, pso_selected_features.astype(bool)], X_img, y)

    # Summary
    print("\nPerformance Summary:")
    print(f"All Features Fitness Score: {baseline_fitness:.4f} | Runtime: {baseline_time:.2f} seconds")
    print(f"PCA Fitness Score: {pca_fitness:.4f} | Runtime: {pca_time:.2f} seconds")
    print(f"GA Fitness Score: {ga_fitness:.4f} | Runtime: {ga_time:.2f} seconds")
    print(f"PSO Fitness Score: {pso_fitness:.4f} | Runtime: {pso_time:.2f} seconds")

if __name__ == "__main__":
    main()