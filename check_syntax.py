#!/usr/bin/env python3
import os
import glob
from py_compile import compile
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s] %(message)s"
)
logger = logging.getLogger(__name__)

def check_syntax():
    tools_dir = os.path.join(os.path.dirname(__file__), "tools")
    py_files = glob.glob(os.path.join(tools_dir, "*.py"))
    
    for py_file in py_files:
        logger.info(f"Checking syntax for {py_file}")
        try:
            compile(py_file, doraise=True)
            logger.info(f"Syntax check passed for {py_file}")
        except Exception as e:
            logger.error(f"Syntax error in {py_file}: {str(e)}")

if __name__ == "__main__":
    check_syntax()