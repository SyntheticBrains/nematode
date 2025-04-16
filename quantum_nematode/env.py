GRID_SIZE = 5
START_POS = (1, 1)
FOOD_POS = (GRID_SIZE - 1, GRID_SIZE - 1)


class MazeEnvironment:
    def __init__(self):
        self.agent_pos = list(START_POS)
        self.goal = FOOD_POS

    def get_state(self):
        dx = self.goal[0] - self.agent_pos[0]
        dy = self.goal[1] - self.agent_pos[1]

        # Debug: Print the current state
        print(f"Agent position: {self.agent_pos}, Goal: {self.goal}, dx={dx}, dy={dy}")
        return dx, dy

    def move_agent(self, action):
        print(f"Action received: {action}, Current position: {self.agent_pos}")  # Debug

        if action == "up" and self.agent_pos[1] < GRID_SIZE - 1:
            self.agent_pos[1] += 1
        elif action == "down" and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == "right" and self.agent_pos[0] < GRID_SIZE - 1:
            self.agent_pos[0] += 1
        elif action == "left" and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        else:
            print(f"Invalid action: {action}, staying in place.")  # Debug

        print(f"New position: {self.agent_pos}")  # Debug

    def reached_goal(self):
        return tuple(self.agent_pos) == self.goal

    def render(self):
        grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        grid[self.goal[1]][self.goal[0]] = "G"  # Mark the goal
        grid[self.agent_pos[1]][self.agent_pos[0]] = "A"  # Mark the agent

        # Print the grid
        print("\n".join(" ".join(row) for row in reversed(grid)))  # Reverse for correct y-axis
        print()  # Add a blank line for spacing
