# This script compares the execution time required to generate histograms of N random samples 
# using two methods: a manual approach and NumPy's built-in np.histogram function. 
# The histograms are computed using 1000 linearly spaced bins, with N varying across 
# the following values: 10, 100, 1000, 10,000, 100,000, and 1,000,000. 
# The output includes both the resulting histograms and a plot illustrating the timing performance.
# Authored by Adrien Bonansea and Samuel Gultom for PHY407 coursework.

# Pseudocode:
# Generate the number of random samples
# Generate the random samples
# For manually-created histogram:
#   Start timing
#   Generate the bins
#   Create count as a list of 0s
#   Loop over each number in the random samples:
#       Loop over each bin:
#           Check if the number is within the bin, if yes:
#               Increment count by 1
#               Stop checking
#   Plot the histogram
#   Stop timing
# For histogram created with np.histogram:
#   Start timing
#   Get the count and binning using the built-in function np.histogram
#   Plot the histogram
#   Stop timing
# Repeat the steps above for all number or random samples
# After getting the execution time for all number of random samples using both method:
# Create an array of number of samples and execution time for both method
# Plot the execution time vs number of random samples from both method in log-log scale

# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Generate the amount of random samples (N = 10, 100, 1000, 10,000, 100,000, and 1,000,000)
N = 10
nums = np.random.randn(N)

# Building histogram manually

# Start timing
start = time()

# Set the amount of bins
bins = np.linspace(-5, 5, num=1001)

# Create lists of zero
counts = []
for i in range(len(bins) - 1):
    counts.append(0)

# Loop through each number and put it in its respective bin
for num in nums:                                    # loop through each number in your data
    i = 0                                           # Initialize position
    while i < len(bins)-1:                          # loop through each bin interval
        if bins[i] <= num < bins[i+1]:              # check if the number falls into the bin
            counts[i] += 1                          # increment the counter for that bin
            break                                   # stop checking once the bin is found
        i += 1

# Plot the manually-created histogram
plt.figure(figsize=(6,6))
plt.bar(bins[:-1], counts, width=0.01)
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Histogram (Created manually)")
plt.savefig(f'Manual {N}')

# Stop timing
end = time()
diff = end - start
print(f"Execution time for {N} samples, manually, is: {diff} sec")

# Building histogram with np.histogram

# Start timing
start = time()

# Use np.histogram and store the counts and bin edges
count2, bins2 = np.histogram(nums, bins=1000, range=(-5,5))

# Plot the histogram created with np.histogram
plt.figure(figsize=(6,6))
plt.bar(bins2[:-1], count2, width=0.01) #We have to use bins2[:-1] as the histogram uses edges and not a full section.
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Histogram (Created using np.histogram)")
plt.savefig(f'Numpy {N}')

# Stop timing
end = time()
diff = end - start
print(f"Execution time for {N} samples, using np.histogram, is: {diff} sec")

# After getting the execution time for all number of random samples using both method, put them as list
nums = np.array([10, 100, 1000, 10000, 100000, 1000000])
time_manual = np.array([0.7340719699859619, 0.7931246757507324, 0.8960528373718262,
                        2.0402190685272217, 12.85228967666626, 158.93219017982483])

time_numpy = np.array([0.6658873558044434, 0.6185874938964844, 0.6872086524963379,
                        0.6471812725067139, 0.6863901615142822, 0.8072061538696289])

# Plot the timing comparison
plt.figure(figsize=(8,6))
plt.loglog(nums, time_manual, marker='o', color='blue', label='Manually')
plt.loglog(nums, time_numpy, marker='o', color='green', label='Using np.histogram')
plt.xlabel("Number of samples (N)")
plt.ylabel("Execution time (seconds)")
plt.title("Execution Time vs Number of Samples")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.savefig('Timing_Plot')