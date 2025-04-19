from quantumnematode.logging_config import (  # pyright: ignore[reportMissingImports]
    logger,
)


def summary(num_runs: int, all_results: list[tuple[int, int, list[tuple]]]) -> None:
    average_steps = sum(steps for _, steps, _ in all_results) / num_runs
    improvement_rate = (
        (all_results[0][1] - all_results[-1][1]) / all_results[0][1] * 100
    )

    if logger.disabled:
        print("All runs completed:")  # noqa: T201
        for run, steps, path in all_results:
            print(f"Run {run}: {steps} steps, Path: {path}")  # noqa: T201

        print("Summary of all runs:")  # noqa: T201
        for run, steps, _path in all_results:
            print(f"Run {run}: {steps} steps")  # noqa: T201

        print(f"Total runs: {len(all_results)}")  # noqa: T201

        print(f"Average steps per run: {average_steps:.2f}")  # noqa: T201
        print(f"Improvement metric (steps): {improvement_rate:.2f}%")  # noqa: T201
    else:
        logger.info("All runs completed:")
        for run, steps, path in all_results:
            logger.info(f"Run {run}: {steps} steps, Path: {path}")

        logger.info("Summary of all runs:")
        for run, steps, _path in all_results:
            logger.info(f"Run {run}: {steps} steps")
        logger.info(f"Total runs: {len(all_results)}")
        logger.info(f"Average steps per run: {average_steps:.2f}")
        logger.info(f"Improvement metric (steps): {improvement_rate:.2f}%")
        logger.info("Simulation completed.")
