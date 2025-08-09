
import time
from typing import TYPE_CHECKING

from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from qiskit_serverless.core.job import Job
import os

IBM_JOB_POLL_INTERVAL = int(os.getenv("IBM_JOB_POLL_INTERVAL", 10))  # Polling interval in seconds
IBM_JOB_TIMEOUT = int(os.getenv("IBM_JOB_TIMEOUT", 60 * 60 * 24))  # Timeout in seconds (24 hours)

def monitor_job(job: "Job", job_description: str = "IBM job") -> None:
    """
    Monitor and log the status of an IBM quantum job.

    Args:
        job: The job object to monitor.
        job_description: Description of the job for logging purposes.
    """
    logger.info(f"{job_description} submitted with Job ID: {job.job_id}")

    # Initial status check
    status = job.status()
    logger.info(f"{job_description} initial status: {status}")

    # Define status descriptions for better logging
    status_descriptions = {
        "QUEUED": "Job is in the Qiskit Function queue",
        "INITIALIZING": "Setting up remote environment and installing dependencies",
        "RUNNING": "Job is running",
        "RUNNING: MAPPING": "Mapping classical inputs to quantum inputs",
        "RUNNING: OPTIMIZING_FOR_HARDWARE": "Optimizing for selected QPU (transpilation, characterization, etc.)",
        "RUNNING: WAITING_FOR_QPU": "Submitted to Qiskit Runtime, waiting in QPU queue",
        "RUNNING: EXECUTING_QPU": "Active Qiskit Runtime job executing on QPU",
        "RUNNING: POST_PROCESSING": "Post-processing results (error mitigation, mapping, etc.)",
        "DONE": "Job completed successfully",
        "ERROR": "Job stopped due to an error",
        "CANCELED": "Job was canceled",
    }

    # Monitor status changes with periodic checks
    last_status = status
    start_time = time.time()

    while status not in ["DONE", "ERROR", "CANCELED"]:
        time.sleep(IBM_JOB_POLL_INTERVAL)
        current_status = job.status()

        # Log status changes
        if current_status != last_status:
            elapsed_time = time.time() - start_time
            description = status_descriptions.get(current_status, current_status)
            logger.info(
                f"{job_description} status changed to: {current_status} - {description} "
                f"(elapsed: {elapsed_time:.1f}s)",
            )
            last_status = current_status

        status = current_status

        # Safety timeout
        if time.time() - start_time > IBM_JOB_TIMEOUT:
            error_message = (
                f"{job_description} monitoring timeout after {IBM_JOB_TIMEOUT} seconds. Last status: {status}"
            )
            logger.warning(error_message)
            raise TimeoutError(error_message)

    # Final status
    final_elapsed = time.time() - start_time
    description = status_descriptions.get(status, status)
    logger.info(
        f"{job_description} final status: {status} - {description} "
        f"(total time: {final_elapsed:.1f}s)",
    )

    if status == "ERROR":
        error_message = (
            f"{job_description} completed with errors. Use job.result() to get error details."
        )
        logger.error(error_message)
        raise RuntimeError(error_message)
    elif status == "CANCELED":
        error_message = f"{job_description} was canceled."
        logger.warning(error_message)
        raise RuntimeError(error_message)
    elif status == "DONE":
        logger.info(f"{job_description} completed successfully.")
