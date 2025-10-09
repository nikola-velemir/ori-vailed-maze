def calculate_discounted_reward(rewards, gamma=0.9):
    g = 0
    for t, r in enumerate(rewards):
        g += (gamma ** t) * r
    return round(g, 3)


def calculate_total_sum_reward(rewards: list[float]):
    return sum(rewards)
