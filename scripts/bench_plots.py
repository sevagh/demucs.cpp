import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Thread counts
    threads = [1, 2, 4, 8, 16, 32]

    # Plain Eigen (no BLAS) data
    no_blas_user_times = [29.08, 33.98, 42.19, 61.25, 109.61, 172.86]
    no_blas_real_times = [38.67, 35.54, 33.09, 32.99, 34.07, 36.32]
    no_blas_memory_gb = [x / (1000 * 1024) for x in [2351168, 2588052, 2587980, 2603056, 2574732, 2574068]]

    # OpenBLAS data
    openblas_user_times = [26.46, 31.24, 38.66, 55.29, 91.14, 177.36]
    openblas_real_times = [34.86, 33.25, 31.66, 30.97, 31.17, 32.23]
    openblas_memory_gb = [x / (1000 * 1024) for x in [1860840, 1861232, 1861756, 1862960, 1872052, 1874976]]

    # AOCL BLIS data
    aocl_blis_user_times = [27.18, 31.43, 38.77, 53.39, 86.28, 170.35]
    aocl_blis_real_times = [34.33, 31.42, 29.63, 28.89, 28.66, 30.26]
    aocl_blis_memory_gb = [x / (1000 * 1024) for x in [1914160, 1915444, 1960084, 1915676, 1961820, 1917904]]

    # Intel MKL data
    mkl_user_times = [26.73, 48.94, 90.06, 173.75, 354.39, 355.57]
    mkl_real_times = [35.29, 33.45, 31.79, 30.88, 31.60, 31.68]
    mkl_memory_gb = [x / (1000 * 1024) for x in [1902768, 1920756, 1912300, 2023360, 2105164, 2106108]]

    # Intel MKL with DEBUG CPU TYPE=5 data
    mkl_debug5_user_times = [27.57, 49.56, 90.10, 176.89, 361.14, 363.09]
    mkl_debug5_real_times = [36.29, 33.74, 32.07, 31.45, 32.36, 32.62]
    mkl_debug5_memory_gb = [x / (1000 * 1024) for x in [1902320, 1921272, 1912340, 2023372, 2105588, 2106064]]

    # Function to create and save a plot
    def create_plot(x, y_data, labels, xlabel, ylabel, title):
        plt.figure(figsize=(7.5, 5))  # Adjust figure size as needed
        for y, label in zip(y_data, labels):
            plt.plot(x, y, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(x)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    # Wall Time (Real Time) Plot
    create_plot(
        threads,
        [no_blas_real_times, openblas_real_times, aocl_blis_real_times, mkl_real_times, mkl_debug5_real_times],
        ['Eigen (No BLAS) Wall Time', 'OpenBLAS Wall Time', 'AOCL BLIS Wall Time', 'MKL Wall Time', 'MKL (CPU_TYPE=5) Wall Time'],
        'Number of Threads', 'Wall Time (seconds)', 'Wall Time Comparison'
    )
    plt.savefig('wall_time_comparison.png')
    plt.close()

    # CPU Time (User Time) Plot
    create_plot(
        threads,
        [no_blas_user_times, openblas_user_times, aocl_blis_user_times, mkl_user_times, mkl_debug5_user_times],  # Replace with user_times data
        ['Eigen (No BLAS) CPU Time', 'OpenBLAS CPU Time', 'AOCL BLIS CPU Time', 'MKL CPU Time', 'MKL (CPU_TYPE=5) CPU Time'],
        'Number of Threads', 'CPU Time (seconds)', 'CPU Time Comparison'
    )
    plt.savefig('cpu_time_comparison.png')
    plt.close()

    # Memory Usage Plot
    create_plot(
        threads,
        [no_blas_memory_gb, openblas_memory_gb, aocl_blis_memory_gb, mkl_memory_gb, mkl_debug5_memory_gb],  # Replace with memory_gb data
        ['Eigen (No BLAS) Memory Usage', 'OpenBLAS Memory Usage', 'AOCL BLIS Memory Usage', 'MKL Memory Usage', 'MKL (CPU_TYPE=5) Memory Usage'],
        'Number of Threads', 'Memory Usage (GB)', 'Memory Usage Comparison'
    )
    plt.savefig('memory_usage_comparison.png')
    plt.close()
