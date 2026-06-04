from kaggle_environments import make

def run_many(n=20):
    for seed in range(n):
        env = make("crawl", configuration={"randomSeed": seed}, debug=True)
        env.run(["main.py", "random"])
        final = env.steps[-1]
        print(f"Seed {seed}:")
        for i, s in enumerate(final):
            print(f"  Player {i}: reward={s.reward}, status={s.status}")

if __name__ == "__main__":
    run_many()