import logging
import resource
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Check if we're running pytest

is_pytest = any("pytest" in arg for arg in sys.argv)

# The former path corresponds to cgroup v2 standard, the latter to cgroup v1
for path in [
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
]:
    if Path(path).exists():
        with open(path) as limit:
            try:
                container_mem = int(limit.read())

                # Use different percentages based on whether we're running pytest
                if is_pytest:
                    # Use 90% for pytest to allow more memory for test execution
                    process_mem = int(container_mem * 0.9)
                    min_mem = 2 * 1024 * 1024 * 1024  # 2GB minimum for pytest
                else:  # Use 80% for normal operations
                    process_mem = int(container_mem * 0.8)
                    min_mem = 1024 * 1024 * 1024  # 1GB minimum

                process_mem = max(process_mem, min_mem)
                resource.setrlimit(resource.RLIMIT_AS, (process_mem, process_mem))

                context = "pytest" if is_pytest else "normal operation"
                log.info(f"RAM limit set to {process_mem:,} bytes for {context}.")
            except ValueError as e:
                log.error(f"Error setting RAM limit: {e}")
                pass
