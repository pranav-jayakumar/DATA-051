#!/usr/bin/env python3

import random

def miguel_sort(arr: list[int]) -> list[int]:
    bools: list[bool] = [False for _ in range(2 ** 32)]
    for x in arr:
        bools[x] = True
    return [i for i, b in enumerate(bools) if b]

if __name__ == "__main__":
    array = []
    for i in range(10):
        x = random.randint(1, 10)
        array.append(x)

    print(array)

    miguel_sort(array)





