#! /usr/bin/env python3

import random

face = lambda x: "Heads" if x == 1 else "Tails"
flip_coin = lambda n: [face(random.randint(0,1)) for _ in range(n)]

coin_flips = 10

result = flip_coin(coin_flips)
heads_count = result.count("Heads")
tails_count = result.count("Tails")

print(f"Heads count: {heads_count}\nTails count: {tails_count}")

