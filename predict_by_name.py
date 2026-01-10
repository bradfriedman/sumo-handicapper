#!/usr/bin/env python3
"""
Wrapper script to run name-based bout predictions from project root.
This allows you to run: python3 predict_by_name.py --interactive
"""
import sys
from src.prediction import predict_by_name

if __name__ == '__main__':
    predict_by_name.main()
