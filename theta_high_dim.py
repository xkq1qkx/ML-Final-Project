import numpy as np
from itertools import product, islice
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load B matrix
with open('B_process_n10.json', 'r') as f:
    B_matrices_json = json.load(f)
    B = np.array(B_matrices_json[-1])
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    print("要计算的B:", B)

# Parameters
batch_size = 10000  # Batch size for processing vectors
dr2 = 1e-5
steps = int(5 // dr2)
print("计算的精度:", dr2, steps)
pic = np.zeros(steps)
n = B.shape[1]

# Generate range for dimensions
range_values = np.arange(-5, 5)

# Total possible vectors
total_vectors = len(range_values) ** n
print(f"Total possible vectors: {total_vectors} (will be processed in batches)")

# Helper function to get batches from itertools.product
def chunked_product(iterable, batch_size):
    """Yield successive n-sized chunks from the iterable"""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, batch_size))  # Take a batch of size `batch_size`
        if not chunk:
            break
        yield np.array(chunk)

# Generate the product and process in batches
product_iter = product(*[range_values] * n)

batch_generator = chunked_product(product_iter, batch_size)

# Process the batches
for batch in tqdm(batch_generator, desc="Processing batches"):
    dots = np.dot(batch, B.T)
    lengths = np.sum(dots**2, axis=1)
    lengths_sorted = np.sort(lengths)
    
    # Count lengths within each radius range
    index = 0
    for j in range(steps):
        while index < len(lengths_sorted) and lengths_sorted[index] <= j * dr2:
            pic[j] += 1
            index += 1

# Cumulative count
for i in range(1, steps):
    pic[i] += pic[i - 1]

# Plot
plt.plot(np.arange(steps) * dr2, pic)
plt.xlabel('R^2')
plt.ylabel('Count')
plt.title('Histogram of Lengths')
plt.show()
