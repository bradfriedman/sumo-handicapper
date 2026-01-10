#!/usr/bin/env python3
"""
Wrapper script to run bout predictions from project root.
This allows you to run: python3 predict_bouts.py --interactive
"""
import sys
from src.prediction import predict_bouts

if __name__ == '__main__':
    predict_bouts.main()
