import tinypywasm
# pywasm.on_debug()


def _fib(n):
    if n <= 1:
        return n
    return _fib(n - 1) + _fib(n - 2)


def fib(_: tinypywasm.Store, n: int):
    return _fib(n)


runtime = tinypywasm.load('./examples/env.wasm', {'env': {'fib': fib}})
r = runtime.exec('get', [10])
print(r)  # 55
