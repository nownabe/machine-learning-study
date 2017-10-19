from scipy.stats import chi2_contingency

data = [
    [950, 348],
    [117, 54],
]
print(chi2_contingency(data))
# (1.4968865615809579, 0.22115103741056918, 1, array([[ 942.79509871,  355.20490129],
#       [ 124.20490129,   46.79509871]]))

# (r - 1)(c - 1) = (2 - 1)(2 - 1) = 1
# x^2{0.05}(1) = 3.84146

# chi2_cont < chi2_{0.05}(1)
# => reject
