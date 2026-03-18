#!/usr/bin/env python3
import sys

"""
乘法数值计算脚本
提供多种乘法运算功能
"""


def multiply_list(numbers):
    """
    列表中的所有数字相乘

    Args:
        numbers (list): 数字列表

    Returns:
        float: 所有数字的乘积

    Raises:
        ValueError: 如果列表为空
    """
    if not numbers:
        raise ValueError("数字列表不能为空")

    result = 1
    for num in numbers:
        result *= num
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python multiply.py <num1> <num2> ...<numn>")
        sys.exit(1)
    nums = []
    for n in sys.argv[1:]:
        nums.append(float(n))
    out = multiply_list(nums)
    print(out)
