#!/usr/bin/env python3

import random

def miguel_sort(arr: list[int]) -> list[int]:
    bools: list[bool] = [False for _ in range(2 ** 32)]
    for x in arr:
        bools[x] = True
    return [i for i, b in enumerate(bools) if b]

def merge(left, right):
    sorted_array = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            sorted_array.append(left[i])
            i += 1
        else:
            sorted_array.append(right[j])
            j += 1

    sorted_array.extend(left[i:])
    sorted_array.extend(right[j:])

    return sorted_array


def mergeSort(arr):
    match arr:
        case []:
            return []
        case [x]:
            return [x]
        case _:
            mid = len(arr) // 2
            left_half = mergeSort(arr[:mid])
            right_half = mergeSort(arr[mid:])
            return merge(left_half, right_half)

if __name__ == "__main__":
    array = []
    for i in range(10):
        x = random.randint(1, 10)
        array.append(x)
    
    n  = len(array)

    print("Original array: ")
    print(array)

    print("Sorted array: ")
    sorted_array = mergeSort(array)
    print(sorted_array)

    # note: miguel sort takes up a LOT of memory. run at your own risk.
    # funny = miguel_sort(array)
    # print(funny)
    






