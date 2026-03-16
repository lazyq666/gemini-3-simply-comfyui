import importlib.util
import logging
import os
import subprocess
import sys


PACKAGE_SPEC = "google.genai"
DISABLE_AUTO_INSTALL_ENV = "GEMINI3_DISABLE_AUTO_INSTALL"
MODULE_DIR = os.path.dirname(__file__)
MODULE_NAME = os.path.basename(MODULE_DIR)
REQUIREMENTS_PATH = os.path.join(MODULE_DIR, "requirements.txt")


def _dependency_installed() -> bool:
    return importlib.util.find_spec(PACKAGE_SPEC) is not None


def _install_requirements() -> bool:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "-r",
        REQUIREMENTS_PATH,
    ]
    logging.info(
        "gemini3 prestartup: missing %s, attempting automatic install with %s",
        PACKAGE_SPEC,
        " ".join(cmd),
    )
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        logging.error("gemini3 prestartup: automatic dependency install failed to start: %s", exc)
        return False

    if result.returncode == 0:
        logging.info("gemini3 prestartup: dependency install completed successfully.")
        if result.stdout.strip():
            logging.info(result.stdout.strip())
        return True

    logging.error("gemini3 prestartup: automatic dependency install failed with exit code %s", result.returncode)
    if result.stderr.strip():
        logging.error(result.stderr.strip())
    if result.stdout.strip():
        logging.error(result.stdout.strip())
    logging.error(
        "gemini3 prestartup: install manually with '%s -m pip install -r custom_nodes/%s/requirements.txt'",
        sys.executable,
        MODULE_NAME,
    )
    return False


def _main() -> None:
    if os.getenv(DISABLE_AUTO_INSTALL_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        logging.info(
            "gemini3 prestartup: automatic dependency install disabled by %s.",
            DISABLE_AUTO_INSTALL_ENV,
        )
        return

    if _dependency_installed():
        return

    if not os.path.exists(REQUIREMENTS_PATH):
        logging.error("gemini3 prestartup: requirements file not found at %s", REQUIREMENTS_PATH)
        return

    _install_requirements()


_main()
