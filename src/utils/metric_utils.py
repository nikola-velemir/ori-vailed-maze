def calculate_discounted_reward(rewards, gamma=0.9):
    G = 0
    for t, r in enumerate(rewards):
        G += (gamma ** t) * r
    return round(G,3)
