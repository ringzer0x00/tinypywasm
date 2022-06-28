# You can limit cycles consumption, if your limit is too low, your work won't be finished when you hit it and an
# "out of gas" exception will be throwed. This is used to ensure that any untrusted code can be stopped alwasys.

import tinypywasm


def instruction_cycle_func(i: tinypywasm.binary.Instruction) -> int:
    if i.opcode == tinypywasm.instruction.i32_add:
        return 1
    if i.opcode == tinypywasm.instruction.i32_mul:
        return 3
    # ... ...
    return 1


option = tinypywasm.Option()
option.instruction_cycle_func = instruction_cycle_func
option.cycle_limit = 100  # a number ge 0, 0 means no limit

runtime = tinypywasm.load('./examples/add.wasm', opts=option)
assert runtime.machine.opts.cycle == 0
r = runtime.exec('add', [4, 5])
assert r == 9  # 4 + 5 = 9
assert runtime.machine.opts.cycle == 4
print(runtime.machine.opts.cycle)
