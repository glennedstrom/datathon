

sum=0
total = 0
for i in range(0,3600):
    for j in range(0,3600):
        if i < j < i+600 or j < i < j+600:
            sum+=1
        total+=1
print(sum/total)
print(sum)
print(total)
