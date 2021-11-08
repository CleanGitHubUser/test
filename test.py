char_list = ['a', 'b', 'b', 'c']

char = ''
for i in char_list:
    char += i

print("char")
print(char)
remain_list = [char]
while remain_list[0] == '':
    new_remain_list = []
    for char in remain_list:
        prosses_list = list(set(char))
        print('prosses_list')
        print(prosses_list)


        for i in prosses_list:
            idx = char.find(i)

            new_remain_list.append(char[:idx] + char[(idx + 1):])
        # remain_list = list(set(remain_list))
        print('remain_list')
        print(remain_list)

