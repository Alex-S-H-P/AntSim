list = [1,2,3,4]

for index , element in enumerate(list):
    if element == 3:
        list = list[:index] + list[index + 1:]
        continue
    print(element)
print(list)