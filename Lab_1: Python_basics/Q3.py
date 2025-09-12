import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc

import numpy as np
import matplotlib.pyplot as plt
import time

#Part A
'''
start = time.time()
'''

N = 100
Bin_edges = np.linspace(-5, 5, num=1000)
list_for_bins = [[] for _ in range(len(Bin_edges)-1)]

nums = np.random.randn(N)

for num in nums:
    for i in range(len(Bin_edges)-1):
        if Bin_edges[i] < num <= Bin_edges[i+1]:
            list_for_bins[i].append(num)
            break

flattened = []
for sublist in list_for_bins:
    for item in sublist:
        flattened.append(item)


plt.figure(figsize=(6,6))
#plt.hist(flattened, bins=Bin_edges, width=0.02)
plt.bar(Bin_edges[:-1], flattened, width=0.02)
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Histogram of Gaussian Numbers using plt.hist()")
plt.show()

'''
end = time.time()
print(end - start)
'''

#Num of nums/ /Time Taken
#10 / / 0.5422592163085938
#100 / / 0.5742769241333008
#1000 / / 1.1014881134033203
#10000 / / 1.7135260105133057
#100000 / / 13.135233163833618
#1000000 / / 135.2844579219818

#Part B part 1
'''
start = time.time()
'''

counts, bin_edges = np.histogram(nums, bins=1000, range=(-5, 5))

plt.figure(figsize=(6,6))
plt.bar(bin_edges[:-1], counts, width=0.02)
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Histogram of Gaussian Numbers using np.histogram")
plt.show()

'''
end = time.time()
print(end - start)
'''
#Num of nums/ /Time Taken
#10 / / 0.5049190521240234
#100 / / 0.49275875091552734
#1000 / / 0.5082061290740967
#10000 / / 0.49869227409362793
#100000 / / 0.5077142715454102
#1000000 / / 0.5557241439819336

#Part B part 2
#Manual bins


nums = np.array([10, 100, 1000, 10000, 100000, 1000000])
time = np.array([0.5422592163085938, 0.5742769241333008,
                       1.1014881134033203, 1.7135260105133057,
                       13.135233163833618, 135.2844579219818])




nums2 = np.array([10, 100, 1000, 10000, 100000, 1000000])
time2 = np.array([0.5049190521240234, 0.49275875091552734,
                       0.5082061290740967, 0.5077142715454102,
                       0.511397123336792, 0.5557241439819336])

plt.figure(figsize=(8,6))
plt.loglog(nums, time, marker='o', color='blue', label='With plt.hist')
plt.xlabel("Number of numbers (N)")
plt.loglog(nums2, time2, marker='o', linestyle='-', color='green', label='With np.histogram')
plt.xlabel("Number of numbers (N)")
plt.ylabel("Time taken (seconds)")
plt.title("Execution Time vs Number of Random Numbers (Fixed Bins)")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.savefig('timing plot')
plt.show()
