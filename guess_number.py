#!/usr/bin/env python3

def miguel_sort(arr: list[int]) -> list[int]:
    bools: list[bool] = [False for _ in range(2 ** 32)]
    for x in arr:
        bools[x] = True
    return [i for i, b in enumerate(bools) if b]


if __name__ == "__main__":
    array = [2, 6, 7, 8, 1, 10, 3, 4, 9, 43]
    sorted_array = miguel_sort(array)
    print(sorted_array)


