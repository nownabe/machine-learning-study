import math

def p(r):
    n = None
    for i in range(r):
        if n == 0:
            next

        if n is None:
            n = 1 - (i/365)
        else:
            n = n * (1 - (i/365))

    return round(1 - n, 4)

for i in [5, 10, 15, 20, 21, 22, 23, 24, 25, 30, 35, 40, 50, 60]:
    print('r = {:>2}, Pr = {:>5}'.format(i, p(i)))
