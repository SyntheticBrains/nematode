import argparse
import logging

from quantumnematode.agent import QuantumNematodeAgent
from quantumnematode.logging_config import logger

# Suppress logs from external libraries like Qiskit
logging.getLogger("qiskit").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Run the Quantum Nematode simulation.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of steps for the simulation (default: 100)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"],
        help="Set the logging level (default: INFO). Use 'NONE' to disable logging.",
    )
    parser.add_argument(
        "--show-last-frame-only",
        action="store_true",
        help="Only display the last frame in the CLI output.",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.log_level == "NONE":
        logger.disabled = True
    else:
        logger.setLevel(args.log_level)

    agent = QuantumNematodeAgent()
    path = agent.run_episode(
        max_steps=args.max_steps, show_last_frame_only=args.show_last_frame_only
    )

    if logger.disabled:
        print("Final path:")
        print(path)
    else:
        logger.info("Simulation completed.")
        logger.info(f"Path taken by the agent: {path}")


if __name__ == "__main__":
    main()
