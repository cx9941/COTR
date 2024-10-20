def calculate_overlap_ratio(list1, list2):
    # 找到重叠项
    overlap = set(list1).intersection(set(list2))
    
    # 计算比率
    overlap_ratio = len(overlap) / min(len(list1), len(list2))
    
    return overlap_ratio

# 示例列表
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

ratio = calculate_overlap_ratio(list1, list2)
print(f"Overlap ratio: {ratio}")