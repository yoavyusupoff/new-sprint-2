import numpy as np
import pandas as pd
from create_graph import run  # Assuming the run function is in existing_file.py

def main():
    # Create a sample DataFrame to simulate input data
    n_rockets = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'x': [10, 20, 30, 40, 50],
        'y': [15, 25, 35, 45, 55],
        'z': [5, 15, 25, 35, 45],
        't': [0, 1, 2, 3, 4]
    })

    # Print the sample DataFrame for verification
    print("Input DataFrame:")
    print(n_rockets)

    # Call the run function with the sample DataFrame
    run(n_rockets)

    # Indicate that the run function has been executed
    print("Run function executed.")

    # Since the run function does not return anything, we assume it performs plotting
    # and other operations internally. If it had a return value, we would print it here.

if __name__ == "__main__":
    main()