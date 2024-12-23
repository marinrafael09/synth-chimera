import pandas as pd

def save_to_excel_with_performance_summary(
    X_num, X_img, y, 
    baseline_fitness, baseline_time, 
    ga_fitness, ga_time, ga_selected_features,
    pso_fitness, pso_time, pso_selected_features,
    filename="data_with_performance_summary.xlsx"
):
    # Converter tensores para NumPy
    X_num_np = X_num.cpu().numpy()
    X_img_np = X_img.cpu().numpy()
    y_np = y.cpu().numpy()

    # Criar DataFrames para os dados
    df_X_num = pd.DataFrame(X_num_np, columns=[f"NumFeature_{i}" for i in range(X_num_np.shape[1])])
    df_X_img = pd.DataFrame(X_img_np.reshape(X_img_np.shape[0], -1), columns=[f"ImgFeature_{i}" for i in range(X_img_np.size // X_img_np.shape[0])])
    df_y = pd.DataFrame(y_np, columns=["Target"])

    # Criar o Performance Summary
    performance_summary = [
        ["Metric", "Value"],
        ["All Features Fitness Score", f"{baseline_fitness:.4f}"],
        ["All Features Runtime (seconds)", f"{baseline_time:.2f}"],
        ["---------------------------------------------------------------------------", ""],
        ["GA Fitness Score", f"{ga_fitness:.4f}"],
        ["GA Runtime (seconds)", f"{ga_time:.2f}"],
        ["GA Selected Features (binary mask)", f"{ga_selected_features}"],
        ["---------------------------------------------------------------------------", ""],
        ["PSO Fitness Score", f"{pso_fitness:.4f}"],
        ["PSO Runtime (seconds)", f"{pso_time:.2f}"],
        ["PSO Selected Features (binary mask)", f"{pso_selected_features}"],
    ]
    df_summary = pd.DataFrame(performance_summary, columns=["Description", "Details"])

    # Combinar os DataFrames em um único arquivo Excel
    with pd.ExcelWriter(filename) as writer:
        df_X_num.to_excel(writer, sheet_name="Numerical Features", index=False)
        df_X_img.to_excel(writer, sheet_name="Image Features", index=False)
        df_y.to_excel(writer, sheet_name="Targets", index=False)
        df_summary.to_excel(writer, sheet_name="Performance Summary", index=False)

    print(f"Data and performance summary saved to {filename}")
