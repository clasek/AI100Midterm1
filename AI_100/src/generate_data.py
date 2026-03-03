import pandas as pd
import random

num_games = 200

data = []

for _ in range(num_games):
    home = random.randint(0, 1)

    # Generate realistic football stats
    points_scored = random.randint(7, 40)
    points_allowed = random.randint(7, 40)
    total_yards = random.randint(250, 450)
    turnovers = random.randint(0, 4)

    win = 1 if points_scored > points_allowed else 0

    data.append([
        points_scored,
        points_allowed,
        total_yards,
        turnovers,
        home,
        win
    ])

df = pd.DataFrame(
    data,
    columns=[
        "points_scored",
        "points_allowed",
        "total_yards",
        "turnovers",
        "home",
        "win"
    ]
)

df.to_csv("data/jets_games.csv", index=False)

print("✅ Dataset generated with", num_games, "games.")