import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
from typing import Dict
from typing import List

import sympy
import torch

import torchinductor

from .. import config
from .. import ir
from ..ir import ReductionHint
from ..utils import free_symbol_startswith
from ..utils import sympy_product
from ..utils import sympy_subs
from ..virtualized import V
from ..virtualized import ops
from .common import DeferredLine
from .common import ExprPrinter
from .common import IndentedBuffer
from .common import Kernel
from .common import OpOverrides
from .common import TIRCSE
from .common import index_prevent_reordering

log = logging.getLogger(__name__)


class TIRPrinter(ExprPrinter):
    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        mod = self.paren(self.doprint(mod))
        if div != "1":
            x = f"({x} / {div})"
        return f"{x} % {mod}"

    def _print_IndexingDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} / {div})"


texpr = TIRPrinter().doprint

dtype_map = {
    torch.float64: "float64",
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def tir_compute_type(dtype):
    tir_type_name = str(dtype).split(".")[-1]
    if tir_type_name == "bool":
        tir_type_name = "int1"
    if tir_type_name in ("float16", "bfloat16"):
        # float16 math is done in float32 inside the kernel
        tir_type_name = "float32"
    return f"T.{tir_type_name}"


def tir_constant(value):
    if value == float("inf"):
        return 'float("inf")'
    elif value == float("-inf"):
        return 'float("-inf")'
    elif math.isnan(value):
        return 'float("nan")'
    return repr(value)


class TIROverrides(OpOverrides):
    """Map element-wise ops to TIR"""

    @staticmethod
    def to_dtype(x, dtype: torch.dtype):
        if dtype == torch.bool:
            return f"({x} != 0)"
        return f"{x}.to({tir_compute_type(dtype)})"

    @staticmethod
    def constant(value, dtype):
        return tir_constant(value)

    @staticmethod
    def abs(x):
        return f"T.abs({x})"

    @staticmethod
    def exp(x):
        return f"T.exp({x})"

    @staticmethod
    def sqrt(x):
        return f"T.sqrt({x})"

    @staticmethod
    def relu(x):
        return f"T.max({x}, T.float32(0))"

    @staticmethod
    def minimum(a, b):
        return f"T.minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"T.max({a}, {b})"

    @staticmethod
    def where(a, b, c):
        # wonkyness to work around https://github.com/openai/triton/issues/532
        # identity calls to force new triton variables (and get access to .shape/.dtype/.numel
        a = ops.identity(a)
        b = ops.identity(b)
        c = ops.identity(c)
        a = ops.identity(
            f"{a} | T.zeros({b}.shape, {a}.dtype) if {b}.numel > 1 else {a}"
        )
        a = ops.identity(
            f"{a} | T.zeros({c}.shape, {a}.dtype) if {c}.numel > 1 else {a}"
        )
        return f"T.where({a}, {b}, {c})"

    @staticmethod
    def cos(x):
        return f"""        
        for i0 in T.serial(10):
            with T.block("T_cos"):
                ax0 = T.axis.spatial(10, i0)
                {x}[ax0] = T.cos(in_ptr0[ax0], dtype="float32")
                """

    @staticmethod
    def sin(x):
        return f"""
        for i0 in T.serial(10):
            with T.block("T_sin"):
                ax0 = T.axis.spatial(10, i0)
                {x}[ax0] = T.sin(in_ptr1[ax0], dtype="float32")
        """

    @staticmethod
    def index_expr(expr, dtype):
        return V.kernel.indexing(expr)[0]

    @staticmethod
    def masked(mask, body, other):
        with V.kernel.mask_loads(mask) as new_mask:
            result = body()
        return ops.where(new_mask, result, TIROverrides.constant(other, torch.float32))

    @staticmethod
    def logical_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def rand(seed, offset, _):  # _ here to keep the contract identical to CPU rand op
        return f"T.rand({seed}, {offset})"

    @staticmethod
    def randn(seed, offset, _):  # _ here to keep the contract identical to CPU randn op
        return f"T.randn({seed}, {offset})"

    @staticmethod
    def rsqrt(x):
        return f"T.rsqrt({x})"

    @staticmethod
    def signbit(x):
        # XX: This is wrong for the value -0.0 in floating point
        return f"T.signbitf({x}) if {x}.dtype is T.float32 else {x} < 0"

    @staticmethod
    def fmod(a, b):
        return f"T.fmod({a}, {b})"

    @staticmethod
    def pow(a, b):
        return f"T.pow({a}, {b})"

    @staticmethod
    def log(x):
        return f"T.log({x}.to(T.float32))"

    @staticmethod
    def isinf(x):
        return f"{x}+1 == {x}"

    @staticmethod
    def isnan(x):
        return f"{x} != {x}"

    @staticmethod
    def round(x):
        return f"T.where({x}<0, {x}-0.5, {x}+0.5).to(T.int32).to(T.float32)"

    @staticmethod
    def floor(x):
        return f"T.floor({x})"

    @staticmethod
    def floordiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Similar to div_floor_kernel_cuda in pytorch core.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        quot = f"{a} // {b}"
        rem = f"{a} % {b}"
        return f"T.where(({a} < 0) != ({b} < 0), T.where({rem} != 0, {quot} - 1, {quot}), {quot})"

    @staticmethod
    def trunc(x):
        return f"{x}.to(T.int32).to(T.float32)"

    @staticmethod
    def truncdiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        return f"{a} // {b}"

    @staticmethod
    def ceil(x):
        return f"T.ceil({x})"


class TIRKernel(Kernel):
    overrides = TIROverrides
    sexpr = texpr

    def __init__(self, reduction_hint=ReductionHint.DEFAULT):
        super(TIRKernel, self).__init__()
        self.cse = TIRCSE(self.newvar_prefix, self.suffix)
        self.iter_vars_count = itertools.count()
        self.body = IndentedBuffer()
        self.indexing_code = IndentedBuffer()
        self.suffix = IndentedBuffer()

    def set_ranges(self, lengths, reduction_lengths):
        self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
        self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
        self.itervars = [sympy.Symbol(f"i{n}") for n in range(len(self.ranges))]
        self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def load(self, name: str, index: sympy.Expr):
        # TODO(shingjan): This may be needed for match_buffer/alloc_buffer
        var = self.args.input(name)
        index = self.rename_indexing(index)
        line = f"{var}[{texpr(index)}]"
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, mode=None):
        # place holder here as TIR doesn't require explicit store
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        line = f"{var}[{texpr(index)}] = {value};"
        self.stores.writeline(name, line)

    def codegen_loops(self, code):
        assert self.itervars
        assert self.ranges

        @dataclasses.dataclass
        class LoopLevel:
            var: sympy.Expr
            size: sympy.Expr

            def lines(self):
                line = f"for {self.var} in T.serial({texpr(self.size)}):"
                for name, value in V.graph.graph_inputs.items():
                    shape = [V.graph.sizevars.size_hint(x) for x in value.get_size()]
                line = f"""for i0 in T.serial({shape[-1]}):
                with T.block("T_relu"):
                    out_ptr0[i0] = {TIROverrides.relu("in_ptr0[i0] + in_ptr1[i0]")}
                """
                return [line]

        loops = [LoopLevel(var, size) for var, size in zip(self.itervars, self.ranges)]
        for loop in loops:
            code.writelines(loop.lines())

    def codegen_body(self):
        """
        Concat output code from index_code, loads, compute, stores,
        suffix into self.body.

        For pointwise kernels, this is called just once at the end.

        For reduction kernels, this generates a loop over the reduction
        axis.
        """

        header = """
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block(\"root\")
        """

        self.body.splice(header)
        self.body.splice(self.indexing_code)
        self.codegen_loops(self.body)
        self.body.splice(self.suffix)
        self.indexing_code.clear()
        self.loads.clear()
        self.stores.clear()
        self.suffix.clear()

    def codegen_kernel(self, name=None):
        code = IndentedBuffer()

        if name is None:
            code.splice(
                f"""
                    import tvm
                    from tvm.script import tir as T
                """
            )

        heuristics_line = f"""
            @tvm.script.ir_module
            class Module:
                @T.prim_func
        """
        code.splice(heuristics_line)

        argdefs, _ = self.args.python_argdefs()
        # TODO(shingjan): ideally this should come from self.size/self.ranges
        argdefs_shape = []
        for _, value in V.graph.graph_inputs.items():
            shape = [V.graph.sizevars.size_hint(x) for x in value.get_size()]
            argdefs_shape.append(shape[0])
        # add an extra one for output
        argdefs_shape.append(argdefs_shape[-1])

        with code.indent():
            def_line = f"def main("
            def_line += ", ".join(
                argdefs[i] + f': T.Buffer[{argdefs_shape[i]}, "float32"]'
                for i in range(len(argdefs) - 1)
            )
            def_line += ") -> None:"
            code.writeline(def_line)
            self.codegen_body()
            with code.indent():
                code.splice(self.body)

        if name is not None:
            return code.getvalue()

        wrapper = IndentedBuffer()
        wrapper.writeline("TIRCodeCache.load('''")
        wrapper.splice(code.getvalue(), strip=True)
        wrapper.writeline('mod = tvm.build(Module, target="llvm --num-cores=16")')
        wrapper.writeline("''').mod")
        print("-------TIR printout-------\n")
        print(wrapper.getvalue())
        print("-------TIR printout-------\n")
        return wrapper.getvalue()

    def call_kernel(self, code, name: str):
        argdefs, call_args = self.args.python_argdefs()
        # TODO(shingjan): constant can be further simplified/folded
        call_args_nd = []
        for idx in range(len(argdefs) - 1):
            line = f"tmp{idx} = tvm.nd.array({call_args[idx]}.numpy())"
            code.writeline(line)
            call_args_nd.append(f"tmp{idx}")
        code.writeline(f"{name}({', '.join(call_args_nd)})")
        # last element in the argument list is always the output
        code.writeline(
            f"{call_args[2]} = torch.from_numpy(tmp{len(argdefs) - 2}.numpy())"
        )


class TIRScheduling:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    @staticmethod
    def can_fuse_horizontal(node1, node2):
        return False

    @classmethod
    def can_fuse_vertical(cls, node1, node2):
        return False

    def codegen_nodes(self, nodes):
        """
        Given a set of pre-fused nodes, generate a TIR kernel.
        """
        scheduler = self.scheduler
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group
        in_suffix = False
        for node in nodes:
            print(node.debug_str_extra())

        with TIRKernel() as kernel:
            vars, reduction_vars = kernel.set_ranges([group], ())

            for node in nodes:
                if node.group[1] in [
                    (group, reduction_group),
                    (group + reduction_group, ()),
                ]:
                    assert not in_suffix
                    node.run(vars, reduction_vars)
                else:
                    in_suffix = True
                    assert node.group[1] == (
                        group,
                        (),
                    ), f"unexpected group: {node.group[1]} != {group}, {reduction_group}"
                    # we can fuse in some extra pointwise into the suffix
                    with kernel.write_to_suffix():
                        node.run(vars, ())

        wrapper = V.graph.wrapper_code
        src_code = kernel.codegen_kernel()
        if src_code in wrapper.kernels:
            kernel_name = wrapper.kernels[src_code]
        else:
            kernel_name = wrapper.next_kernel_name()
            wrapper.kernels[src_code] = kernel_name
            subs_name = kernel_name if config.tvm.ordered_kernel_names else "kernel"
            # src_code = src_code.format(kernel_name=subs_name)
            # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
            # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
            src_code = src_code.replace("#pragma CMT", "#")
            wrapper.define_kernel(kernel_name, src_code)
        kernel.call_kernel(wrapper, kernel_name)
        self.scheduler.free_buffers()

    def flush(self):
        pass
