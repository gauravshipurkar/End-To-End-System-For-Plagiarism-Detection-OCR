import numpy as np

litt = [1, 0.09531714, 0.1352908, 0.09531714, 1,0.3263665,0.1352908, 0.3263665, 1]

ot = sorted(litt, reverse=True)
print(ot)

result = []
for sim in ot:
    result.append(round(sim,2))

print(result)




