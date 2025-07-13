"""IBM Quantum Authenticator."""

import os

from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

IBM_QUANTUM_API_KEY = "IBM_QUANTUM_API_KEY"


class IBMQuantumAuthenticator:
    """Handles authentication with IBM Quantum using an API key."""

    def __init__(self, env_var_token: str = IBM_QUANTUM_API_KEY) -> None:
        load_dotenv()
        self.token = os.environ.get(env_var_token)

    def authenticate(self) -> None:
        """Save the IBM Quantum API token to QiskitRuntimeService for future use."""
        if not self.token:
            error_message = "IBM Quantum API key not found in environment variables."
            raise ValueError(error_message)
        QiskitRuntimeService.save_account(token=self.token, overwrite=True)
