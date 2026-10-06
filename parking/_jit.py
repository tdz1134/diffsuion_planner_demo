# -*- coding: utf-8 -*-
"""可选 numba 加速。

设计: 只把"纯数值 kernel"(输入/输出都是 numpy 数组与标量, 无 dict/list-of-tuple)
交给 numba; A* 主循环(含 heapq/dict)保持纯 Python。这样:
  - 未安装 numba 时 -> `optional_njit` 退化为恒等装饰器, 代码照常运行(慢一些);
  - 安装了 numba 时 -> 同一份 kernel 自动 JIT 提速, 无需改调用处。

因此被 `@optional_njit` 装饰的函数必须写成 numba 兼容(仅用数组/标量/基本控制流)。
"""

try:                                    # pragma: no cover - 取决于环境
    from numba import njit as _njit

    def optional_njit(fn=None, **kwargs):
        kwargs.setdefault("cache", True)
        kwargs.setdefault("fastmath", True)
        if fn is None:
            return lambda f: _njit(f, **kwargs)
        return _njit(fn, **kwargs)

    NUMBA_AVAILABLE = True
except Exception:                       # numba 未安装 -> 恒等装饰器
    NUMBA_AVAILABLE = False

    def optional_njit(fn=None, **kwargs):
        if fn is None:
            return lambda f: f
        return fn
