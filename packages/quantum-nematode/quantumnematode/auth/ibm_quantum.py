# pragma: no cover

"""IBM Quantum Authenticator."""

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from quantumnematode.errors import (
    ERROR_MISSING_IMPORT_QISKIT_IBM_CATALOG,
)

if TYPE_CHECKING:
    from qiskit_ibm_catalog import QiskitFunctionsCatalog

ERROR_IBM_QUANTUM_API_KEY_NOT_FOUND = (
    "IBM Quantum API key not found in environment variables. "
    "Set the 'IBM_QUANTUM_API_KEY' environment variable."
)
ERROR_IBM_QUANTUM_CHANNEL_NOT_SET = (
    "IBM Quantum channel not set in environment variables. "
    "Set the 'IBM_QUANTUM_CHANNEL' environment variable."
)
ERROR_IBM_QUANTUM_CRN_NOT_SET = (
    "IBM Quantum CRN not set in environment variables. "
    "Set the 'IBM_QUANTUM_CRN' environment variable."
)


class IBMQuantumAuthenticator:
    """Handles authentication with IBM Quantum using an API key."""

    def __init__(self) -> None:
        load_dotenv()
        self.channel = os.environ.get("IBM_QUANTUM_CHANNEL")
        self.crn = os.environ.get("IBM_QUANTUM_CRN")
        self.token = os.environ.get("IBM_QUANTUM_API_KEY")

    def authenticate_runtime_service(self) -> None:
        """Save the IBM Quantum API token to QiskitRuntimeService for future use."""
        if not self.token:
            error_message = ERROR_IBM_QUANTUM_API_KEY_NOT_FOUND
            raise ValueError(error_message)

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as err:
            error_message = (
                "IBM Quantum support requires the 'qiskit_ibm_runtime' package. "
                "Install it with the extra `qpu`."
            )
            raise ImportError(error_message) from err

        QiskitRuntimeService.save_account(token=self.token, overwrite=True)

    def get_functions_catalog(self) -> "QiskitFunctionsCatalog":
        """Get the QiskitFunctionsCatalog instance for IBM Quantum."""
        if not self.token:
            error_message = ERROR_IBM_QUANTUM_API_KEY_NOT_FOUND
            raise ValueError(error_message)

        if not self.channel:
            error_message = ERROR_IBM_QUANTUM_CHANNEL_NOT_SET
            raise ValueError(error_message)

        if not self.crn:
            error_message = ERROR_IBM_QUANTUM_CRN_NOT_SET
            raise ValueError(error_message)

        try:
            from qiskit_ibm_catalog import QiskitFunctionsCatalog
        except ImportError as err:
            error_message = ERROR_MISSING_IMPORT_QISKIT_IBM_CATALOG
            raise ImportError(error_message) from err

        return QiskitFunctionsCatalog(
            channel=self.channel,
            instance=self.crn,
            token=self.token,
        )
