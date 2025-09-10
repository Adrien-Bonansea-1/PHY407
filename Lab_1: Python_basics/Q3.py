import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc

import numpy as np
import matplotlib.pyplot as plt
import time

#Part A
'''
start = time.time()
N = 1000000
Bin_edges = np.linspace(-5, 5, num=1001)
list_for_bins = [[] for _ in range(len(Bin_edges)-1)]

nums = np.random.randn(N)

for num in nums:
    for i in range(len(Bin_edges)-1):
        if Bin_edges[i] < num <= Bin_edges[i+1]:
            list_for_bins[i].append(num)
            break

# Flatten bins back into one list for plotting
flattened = [item for sublist in list_for_bins for item in sublist]

plt.hist(flattened, bins=Bin_edges)
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Histogram of Gaussian Numbers")
plt.show()

end = time.time()
print(end - start)
'''

#Num of nums/ /Time Taken
#10 / / 0.6175589561462402
#100 / / 0.7648630142211914
#1000 / / 0.9234230518341064
#10000 / / 1.7492051124572754
#100000 / / 12.414344072341919
#1000000 / / 127.03001189231873

#Part B part 1

'''
start = time.time()
N = 100
nums = np.random.randn(N)
plt.hist(nums, bins=1000, range=(-5,5))
plt.show()
end = time.time()
print(end - start)
'''

#Num of nums/ /Time Taken
#10 / / 0.5049190521240234
#100 / / 0.49275875091552734
#1000 / / 0.5082061290740967
#10000 / / 0.5077142715454102
#100000 / / 0.511397123336792
#1000000 / / 0.5557241439819336

#Part B part 2
#Manual bins
nums = np.array([10, 100, 1000, 10000, 100000, 1000000])
time = np.array([0.6175589561462402, 0.7648630142211914,
                       0.9234230518341064, 1.7492051124572754,
                       12.414344072341919, 127.03001189231873])

#GBT was used for titles and labels
plt.figure(figsize=(8,6))
plt.loglog(nums, time, marker='o', linestyle='-', color='blue')
plt.xlabel("Number of numbers (N)")
plt.ylabel("Time taken (seconds)")
plt.title("Execution Time vs Number of Random Numbers")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

#With hist bins
nums = np.array([10, 100, 1000, 10000, 100000, 1000000])
time = np.array([0.5049190521240234, 0.49275875091552734,
                       0.5082061290740967, 0.5077142715454102,
                       0.511397123336792, 0.5557241439819336])

plt.figure(figsize=(8,6))
plt.scatter(nums, time, marker='o', linestyle='-', color='green')
plt.xlabel("Number of numbers (N)")
plt.ylabel("Time taken (seconds)")
plt.ylim(0, 1)
plt.title("Execution Time vs Number of Random Numbers (Fixed Bins)")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()


