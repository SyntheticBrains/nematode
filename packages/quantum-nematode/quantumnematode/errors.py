"""Define error messages for the Quantum Nematode package."""

ERROR_MISSING_IMPORT_QISKIT_IBM_RUNTIME = (
    "IBM Quantum support requires the 'qiskit_ibm_runtime' package. "
    "Install it with the extra `qpu`."
)
ERROR_MISSING_IMPORT_QISKIT_IBM_CATALOG = (
    "IBM Quantum support requires the 'qiskit_ibm_catalog' package. "
    "Install it with the extra `qpu`."
)
ERROR_MISSING_IMPORT_QISKIT_AER = (
    "QPU simulation support requires the 'qiskit_aer' package. "
    "Install it with the extra `cpu` or `qpu`."
)
