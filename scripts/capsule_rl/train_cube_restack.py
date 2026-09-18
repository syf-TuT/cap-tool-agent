"""Train privileged high-level Cube Restack using the shared Cube Stack Capsule recipe."""

from scripts.capsule_rl.train_cube_stack import main

if __name__ == "__main__":
    main(task="cube_restack")
