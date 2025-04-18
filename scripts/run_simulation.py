import sys

sys.path.append("..")

from quantumnematode.agent import QuantumNematodeAgent
from quantumnematode.logging_config import logger


def main():
    agent = QuantumNematodeAgent()
    path = agent.run_episode(max_steps=10)

    print("Final path:")
    print(path)


if __name__ == "__main__":
    main()
