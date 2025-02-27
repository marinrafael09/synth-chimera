import time, datetime
import torch
from generate_dataset import generate_multimodal_dataset
from cnn_fitness import evaluate_features
from optimization import genetic_algorithm, particle_swarm_optimization, pca_feature_selection
from device_detection import get_available_device
from save_dataset import save_to_excel_with_performance_summary

def main():
    # Detect device
    device = get_available_device()
    print(f"Using device: {device}")
    
    # Generate dataset
    n_samples: int = int(input("Number of Samples: "))
    n_features: int = int(input("Number of Features (Even number - Except Image): "))
    n_classes: int = int(input("Number of Classes: "))

    num_pop: int = int(input("Number of Population (GA): "))
    num_gen: int = int(input("Number of Generations (GA): "))
    
    num_part: int = int(input("Number of Particle (PSO): "))
    num_iter: int = int(input("Number of Iteractions (PSO): "))

    print('Generating multimodal dataset')
    X_num, X_img, y = generate_multimodal_dataset(num_samples=n_samples, num_features=n_features, image_size=(64,64), num_classes=n_classes)
    X_num, X_img, y = X_num.to(device), X_img.to(device), y.to(device)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # filename = f"dataset_{timestamp}.xlsx"

    # save_to_excel_with_performance_summary(
    #     X_num, X_img, y,
    #     baseline_fitness=0, baseline_time=0,
    #     ga_fitness=0, ga_time=0, ga_selected_features=0,
    #     pso_fitness=0, pso_time=0, pso_selected_features=0,
    #     filename=filename
    # )
    
    # Fitness Function 
    fitness_fn = lambda X_selected, X_img, y, use_image : evaluate_features(X_selected, X_img, y, device, use_image)

    # Baseline Evaluation (All Features)
    print("\nEvaluating all features (baseline)...")
    start_time = time.time()
    baseline_fitness = fitness_fn(X_num, X_img, y, True)
    baseline_time = time.time() - start_time

    # PCA Feature Selection
    # print("\nEvaluating PCA-selected features...")
    # start_time = time.time()
    # X_pca, pca_selected_features = pca_feature_selection(X_num, n_components=5)
    # X_pca = X_pca.to(device)
    # pca_time = time.time() - start_time
    # pca_fitness = fitness_fn(X_pca, X_img, y)
   
    # GA Feature Selection
    print("\n Evaluating GA-selected features...")
    start_time = time.time()
    ga_selected_features = genetic_algorithm(X_num, X_img, y, fitness_fn, device=device, num_generations=num_gen, population_size=num_pop)
    ga_time = time.time() - start_time
    ga_fitness = fitness_fn(X_num[:, ga_selected_features[:-1].astype(bool)], X_img, y, ga_selected_features[-1].astype(bool))

    # PSO Feature Selection
    print("\n Evaluating PSO-selected features...")
    start_time = time.time()
    pso_selected_features = particle_swarm_optimization(X_num, X_img, y, fitness_fn, device=device, num_iterations=num_iter, num_particles=num_part)
    pso_time = time.time() - start_time
    pso_fitness = fitness_fn(X_num[:, pso_selected_features[:-1].astype(bool)], X_img, y, pso_selected_features[-1].astype(bool))

    # Summary
    print("\nPerformance Summary:")
    print("---------------------------------------------------------------------------")
    print(f"Number of Samples: {n_samples}")
    print(f"Number of Features (Even number - Except Image): {n_features}")
    print(f"Number of Classes: {n_classes}")
    print("---------------------------------------------------------------------------")  
    print(f"All Features Fitness Score: {baseline_fitness:.4f} | Runtime: {baseline_time:.2f} seconds")
    print("---------------------------------------------------------------------------")
    # print(f"PCA Fitness Score: {pca_fitness:.4f} | Runtime: {pca_time:.2f} seconds")
    # print(f"PCA-selected features (binary mask): {pca_selected_features}")
    # print("---------------------------------------------------------------------------")
    print(f"Number of Population (GA): {num_pop}")
    print(f"Number of Generations (GA): {num_gen}")
    print(f"GA Fitness Score: {ga_fitness:.4f} | Runtime: {ga_time:.2f} seconds")
    print(f"GA-selected features (binary mask): {ga_selected_features}")
    print("---------------------------------------------------------------------------")
    print(f"Number of Particle (PSO): {num_part}")
    print(f"Number of Iteractions (PSO): {num_iter}")
    print(f"PSO Fitness Score: {pso_fitness:.4f} | Runtime: {pso_time:.2f} seconds")
    print(f"PSO-selected features (binary mask): {pso_selected_features}")

    # filename = f"summary_{timestamp}.xlsx"

    # save_to_excel_with_performance_summary(
    #     X_num, X_img, y,
    #     baseline_fitness=baseline_fitness, baseline_time=baseline_time,
    #     ga_fitness=ga_fitness, ga_time=ga_time, ga_selected_features=ga_selected_features,
    #     pso_fitness=pso_fitness, pso_time=pso_time, pso_selected_features=pso_selected_features,
    #     filename=filename
    # )
    
if __name__ == "__main__":
    main()
