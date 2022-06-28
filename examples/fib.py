import tinypywasm
# pywasm.on_debug()

runtime = tinypywasm.load('./examples/fib.wasm')
r = runtime.exec('fib', [10])
print(r)
