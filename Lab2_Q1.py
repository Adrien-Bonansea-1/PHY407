
import numpy as np

#Part A: Pseudo code is in the PDF file

#Part (B)
print("Part B---------------")
data = np.loadtxt("cdata.txt")

#Correct answer
correct_answer = np.std(data, ddof = 1)
print("The assumed correct output from np.std is = ", correct_answer)


# form 1: 2 pass method
def std_form_1(data):
    x_bar = np.mean(data)  # calculate mean, np.mean is doing the 1st pass

    temp_sum = 0
    for xi in data:  # Caclulating the term inside the sum, doing the 2nd pass
        temp_sum += (xi - x_bar) ** 2

    sigma1_squared = temp_sum / ((len(data)) - 1)  # Doing the rest of the equation to find sigma and return it
    sigma1 = np.sqrt(sigma1_squared)
    return sigma1


ans1 = std_form_1(data)
print("The standard diviation of the 2 pass method is = ", ans1)
# dif2 = np.abs(correct_answer - ans1)/ans1
dif1 = np.abs(ans1 - correct_answer) / correct_answer  # Relative error
print("The relative error of the 2 pass method is = ", np.abs(dif1))


# form 2: 1 pass method
def std_form_2(data):
    x_bar = 0  # Initialize temporary storage variables
    temp_sum = 0

    n = (len(data))
    for xi in data:  # 1st and only pass
        temp_sum += xi ** 2  # squared values
        x_bar += xi  # non_squared values

    x_mean = x_bar / n  # calculate mean
    top_term = temp_sum - (n * (x_mean ** 2))  # top term in the 1 pass formula

    sigma2_squared = top_term / ((len(data) - 1))
    sigma2 = np.sqrt(sigma2_squared)
    return (sigma2)  # Return 1 pass method for any data set


ans2 = std_form_2(data)
print("The standard diviation of the 1 pass method is = ", ans1)
# dif2 = np.abs(correct_answer - ans2)/ans2
dif2 = np.abs(ans2 - correct_answer) / correct_answer  # Relative error
print("The relative error of the 1 pass method is = ", np.abs(dif2))

#Part C
print("Part C---------------")
#Load data sets for 0 mean-centred and 1e7 mean-centred
list_vals_1 = np.random.normal(0, 1, 2000)
list_vals_2 = np.random.normal(1e7, 1, 2000)

#0 centered
print("0 mean centered")
print(f"with np.std: {np.std(list_vals_1)}")
print("___")
print(f"with 1st std: {std_form_1(list_vals_1)}")
print(f"dif to np.std = : {np.abs(std_form_1(list_vals_1) - np.std(list_vals_1))/(np.std(list_vals_1))}")
print("___")
print(f"with 2nd std: {std_form_2(list_vals_1)}")
print(f"dif to np.std = : {np.abs(std_form_2(list_vals_1) - np.std(list_vals_1))/(np.std(list_vals_1))}")
print("______________")
print("______________")
#1e7 centered
print("1e7 mean centered")
print(f"with np.std: {np.std(list_vals_2)}")
print("___")
print(f"with 1st std: {std_form_1(list_vals_2)}")
print(f"dif to np.std = : {np.abs(std_form_1(list_vals_2) - np.std(list_vals_2))/(np.std(list_vals_2))}")
print("___")
print(f"with 2nd std: {std_form_2(list_vals_2)}")
print(f"dif to np.std = : {np.abs(std_form_2(list_vals_2) - np.std(list_vals_2))/(np.std(list_vals_2))}")

#Part D
print("Part D---------------")
# Edited 1 pass method from earlier
def std_form_2_edited(data):
    # Initialize variables
    total = 0
    temp_sum = 0
    num = 0

    # start iterating over data, still done in 1 pass
    for xi in data:
        total += xi
        num += 1
        cur_mean = total/num # recalculate a mean with every additional value
        temp_sum += (xi - cur_mean) ** 2  # remove current mean from current value, then square it

    sigma2_squared = temp_sum / ((len(data) - 1))  # calculate the rest of the equation like in the unedited version
    sigma2 = np.sqrt(sigma2_squared)
    return (sigma2)


ans2 = std_form_2_edited(data)
print(ans2)

# relative error
print(f"with 2nd std: {std_form_2_edited(list_vals_2)}")
print(f"dif to np.std = : {np.abs(std_form_2_edited(list_vals_2) - np.std(list_vals_2)) / (np.std(list_vals_2))}")


