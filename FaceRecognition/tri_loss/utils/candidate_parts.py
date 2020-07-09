import copy
def combine_2_parts(p1,p2):
    if p1[1]<p2[0]-0.0001:
        parts=[p1,p2]
    elif p2[1]<p1[0]-0.0001:
        parts=[p2,p1]
    else:
        parts=[(min(p1[0],p2[0]),max(p2[1],p1[1]))]

    return parts


def combine_parts(ps):
    ps1=[]
    for i in range(len(ps)):
        temp=[ps[i]]
        for j in range(len(ps)):
            if j<=i:
                continue
            for k in temp:
                temp=temp+combine_2_parts(k,ps[j])

        ps1+=temp

    return ps1

def is_point_part(s,ps):

    for temp in ps:
        if s>=temp[0]-0.0001 and s<=temp[1]+0.0001:
            return True
    return False

def is_same_part(p1,p2):
    idx_in=0
    idx_or=0
    for i in range(100):
        s=0.01*i
        if is_point_part(s,p1) and is_point_part(s,p2):
            idx_in+=1
        if is_point_part(s,p1) or is_point_part(s,p2):
            idx_or+=1
    return idx_in==idx_or

def _combinators(_handle, items, n):
    ''' 抽取下列组合的通用结构'''
    if n == 0:
        yield [ ]
    for i, item in enumerate(items):
        this_one = [item]
        for cc in _combinators(_handle, _handle(items, i), n-1):
            yield this_one + cc

def combinations(items, n):
    def skipIthItem(items, i):
        return items[:i] + items[i + 1:]
    return _combinators(skipIthItem, items, n)

def uniqueCombinations(items, n):
    '''取得n个不同的项，顺序无关'''
    def afterIthItem(items, i):
        return items[i+1:]
    return _combinators(afterIthItem, items, n)



def generate_parts(meta_parts=[(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1), (0, 1 / 4), (1 / 4, 2 / 4), (2 / 4, 3 / 4), (3 / 4, 1)]):
    ori_parts = []
    for nums in range(len(meta_parts)):
        ori_parts += [temp for temp in uniqueCombinations(meta_parts, nums + 1)]
        # print([temp for temp in uniqueCombinations(meta_parts, nums + 1)])
        # print(len([temp for temp in uniqueCombinations(meta_parts, nums + 1)]))
        # print("*" * 20)

    all_parts = []
    for temp in ori_parts:
        is_exsit = False
        for i in range(len(all_parts)):
            if is_same_part(temp, all_parts[i]) == True:
                # print(temp, all_parts[i])
                # print('x' * 20)
                is_exsit = True
                break
        if is_exsit == False:
            all_parts.append(temp)
    return  all_parts


# meta_parts = [(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1), (0, 1 / 4), (1 / 4, 2 / 4), (2 / 4, 3 / 4), (3 / 4, 1)]






# # all_parts_iter=uniqueCombinations(meta_parts,2)
# #
# #
# # ori_parts=[temp for temp in all_parts_iter]+[temp for temp in uniqueCombinations(meta_parts,1)]
#
#
#
#
#
# all_parts=[]
# for temp in ori_parts:
#     is_exsit=False
#     for i in range(len(all_parts)):
#         if is_same_part(temp,all_parts[i])==True:
#             print(temp,all_parts[i])
#             print('x'*20)
#             is_exsit=True
#             break
#     if is_exsit==False:
#         all_parts.append(temp)
#
#
#
#
#
#
#
#
#
#
#
#
# print(all_parts)
# print(len(all_parts))
#
# print("="*30)
#
# print(ori_parts)
# print(len(ori_parts))

# def generate_parts(s,k):
#
#
#     num_parts=2
#
#     parts=[]
#     for i in range(num_parts):
#         for j in range(len(meta_parts)):
#             if meta_parts[j] not in parts:
#                 parts.append(meta_parts[j])
#                 break
#     _parts=combine_parts()
#
#
#
#
#     for i in range(len(meta_parts)):
#         for j in range
if __name__ == '__main__':
    meta_parts = [(0, 1 / 6),(1 / 6, 2 / 6), (2 / 6, 3 / 6),(3 / 6, 4 / 6),(4 / 6, 5 / 6),(5 / 6, 1), (0, 1 / 4), (1 / 4, 2 / 4), (2 / 4, 3 / 4), (3 / 4, 1)]
    all_parts=generate_parts(meta_parts)
    print(all_parts)
    print(len(all_parts))


