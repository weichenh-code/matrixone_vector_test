#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 cuVS brute_force API
"""

import sys
import inspect

print("=" * 70)
print("检查 cuVS brute_force 模块 API")
print("=" * 70)

try:
    from cuvs.neighbors import brute_force
    print(f"✓ 成功导入 cuvs.neighbors.brute_force")
    print()
    
    # 列出所有可用的属性和方法
    print("brute_force 模块中的所有属性:")
    print("-" * 70)
    attrs = dir(brute_force)
    for attr in sorted(attrs):
        if not attr.startswith('_'):
            obj = getattr(brute_force, attr)
            print(f"  {attr}: {type(obj).__name__}")
    print()
    
    # 检查 build 函数
    if hasattr(brute_force, 'build'):
        print("brute_force.build 函数签名:")
        print("-" * 70)
        try:
            sig = inspect.signature(brute_force.build)
            print(f"  {sig}")
        except:
            print("  无法获取签名")
        print()
    
    # 检查 search 函数
    if hasattr(brute_force, 'search'):
        print("brute_force.search 函数签名:")
        print("-" * 70)
        try:
            sig = inspect.signature(brute_force.search)
            print(f"  {sig}")
        except:
            print("  无法获取签名")
        print()
    
    # 检查是否有 SearchParams
    if hasattr(brute_force, 'SearchParams'):
        print("✓ brute_force.SearchParams 存在")
        print(f"  类型: {type(brute_force.SearchParams)}")
    else:
        print("✗ brute_force.SearchParams 不存在")
        print("  可能的替代:")
        for attr in attrs:
            if 'param' in attr.lower() or 'search' in attr.lower():
                print(f"    - {attr}")
    print()
    
    # 尝试查看文档
    print("brute_force.search 文档:")
    print("-" * 70)
    if hasattr(brute_force, 'search'):
        doc = brute_force.search.__doc__
        if doc:
            print(doc)
        else:
            print("  无文档字符串")
    print()
    
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 70)
