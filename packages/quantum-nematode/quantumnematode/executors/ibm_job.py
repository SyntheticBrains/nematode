"""IBM quantum job monitoring with retry logic for network connectivity issues."""

import time
from typing import TYPE_CHECKING

from quantumnematode.executors.ibm_job_status import IBMJobStatus, IBMJobStatusInfo
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from qiskit_serverless.core.job import Job
import os

# Constants for network retry handling
MAX_RETRY_ATTEMPTS = int(os.getenv("IBM_JOB_MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.getenv("IBM_JOB_RETRY_BASE_DELAY", "30.0"))  # Backoff base in seconds

POLL_INTERVAL = int(os.getenv("IBM_JOB_POLL_INTERVAL", "10"))  # Polling interval in seconds
TIMEOUT = int(os.getenv("IBM_JOB_TIMEOUT", str(60 * 60 * 24)))  # Timeout in seconds


def _get_job_status_with_retry(job: "Job", job_description: str = "IBM job") -> str:
    """
    Get job status with retry logic for network connectivity issues.

    Args:
    ----
        job: The job object to get status from.
        job_description: Description of the job for logging purposes.

    Returns
    -------
        The job status string.

    Raises
    ------
        RuntimeError: If all retry attempts fail.

    """
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            return job.status()
        except Exception as e:
            # Check if this is a network connectivity issue
            error_str = str(e)
            is_network_error = (
                "QiskitServerlessException" in str(type(e))
                or "AUTH1001" in error_str
                or "connection" in error_str.lower()
                or "network" in error_str.lower()
                or "timeout" in error_str.lower()
            )

            if is_network_error and attempt < MAX_RETRY_ATTEMPTS - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"{job_description} status check failed (attempt {attempt + 1}/"
                    f"{MAX_RETRY_ATTEMPTS}): {e}. Retrying in {delay:.1f}s...",
                )
                time.sleep(delay)
                continue

            # Either not a network error or exhausted retries
            if is_network_error:
                error_message = (
                    f"{job_description} status check failed after "
                    f"{MAX_RETRY_ATTEMPTS} attempts: {e}"
                )
                logger.error(error_message)
                raise RuntimeError(error_message) from e

            # Re-raise non-network errors immediately
            raise

    # This should never be reached, but included for completeness
    error_message = f"Unexpected error in status retry logic for {job_description}"
    raise RuntimeError(error_message)


def monitor_job(job: "Job", job_description: str = "IBM job") -> None:
    """
    Monitor and log the status of an IBM quantum job.

    Args:
        job: The job object to monitor.
        job_description: Description of the job for logging purposes.
    """
    logger.info(f"{job_description} submitted with Job ID: {job.job_id}")

    # Initial status check with retry logic
    status = _get_job_status_with_retry(job, job_description)
    logger.info(f"{job_description} initial status: {status}")

    # Monitor status changes with periodic checks
    last_status = status
    start_time = time.time()

    while not IBMJobStatus.is_terminal(status):
        time.sleep(POLL_INTERVAL)
        current_status = _get_job_status_with_retry(job, job_description)

        # Log status changes
        if current_status != last_status:
            elapsed_time = time.time() - start_time
            description = IBMJobStatusInfo.get_description(current_status)
            logger.info(
                f"{job_description} status changed to: {current_status} - {description} "
                f"(elapsed: {elapsed_time:.1f}s)",
            )
            last_status = current_status

        status = current_status

        # Safety timeout
        if time.time() - start_time > TIMEOUT:
            error_message = (
                f"{job_description} monitoring timeout after {TIMEOUT} seconds. "
                f"Last status: {status}"
            )
            logger.warning(error_message)
            raise TimeoutError(error_message)

    # Final status
    final_elapsed = time.time() - start_time
    description = IBMJobStatusInfo.get_description(status)
    logger.info(
        f"{job_description} final status: {status} - {description} "
        f"(total time: {final_elapsed:.1f}s)",
    )

    if status == IBMJobStatus.ERROR:
        error_message = (
            f"{job_description} completed with errors. Use job.result() to get error details."
        )
        logger.error(error_message)
        raise RuntimeError(error_message)
    if status in (IBMJobStatus.CANCELED, IBMJobStatus.CANCELLED):
        error_message = f"{job_description} was canceled."
        logger.warning(error_message)
        raise RuntimeError(error_message)
    if status == IBMJobStatus.DONE:
        logger.info(f"{job_description} completed successfully.")
