# TinyPYWasm

Based on [pywasm](https://github.com/mohanson/pywasm)

Interpreter for TinyWASM, a (Turing complete) subset of WebAssembly written in pure Python for experimental purposes.
It supports signed i32 __ONLY__.

# License

[MIT](./LICENSE)

# TinyWASM syntax:

```
module ::= (module type* func* table)`
    type ::= (type $tname (func ft))
    bt, ft ::= t* → t*
    t ::= i32 
    func ::= (func $fname $tname) 
            | (func $fname $tname (param t*) (result t*) instr*) 
    memory ::= (memory ns)
                | (memory ns nmax)
    table ::= (table n*) 
    instr ::= data | control 
        data ::=  t.const n 
                | t.binop 
                | t.unop
                | local.get $var 
                | local.set $var
                | global.get $var 
                | global.set $var 
                | load 
                | store 
        control ::= block bt instr* end 
                | loop bt instr* end
                | call ft 
                | call_indirect ft 
                | br_if l
                | return

n, l, tidx, ns, nmax, bt, ft ::= number
$var, $fname, $tname ::= label

```
