import tvm
from tvm import relay
from tvm.script import tir as T
from tvm.relay import expr as _expr
from tvm.relay import op as _op
from tvm.contrib import graph_executor
from tvm import meta_schedule as ms

import torch

# fmt: off
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(p0: T.Buffer[(64, 3, 7, 7), "float32"], p1: T.Buffer[(8, 3, 224, 224), "float32"], out_ptr0: T.Buffer[(8, 64, 112, 112), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        T_layout_trans = T.alloc_buffer([16, 1, 7, 7, 3, 4], dtype="float32")
        T_layout_trans_1 = T.alloc_buffer([8, 1, 224, 224, 3], dtype="float32")
        data_pad = T.alloc_buffer([8, 1, 230, 230, 3], dtype="float32")
        conv2d_NCHWc = T.alloc_buffer([8, 16, 112, 112, 4], dtype="float32")
        conv2d_NCHWc_global = T.alloc_buffer([8, 16, 112, 112, 4], dtype="float32")
        for i0_i1_i2_fused in T.parallel(1840, annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 224):
                for ax4_fused in T.vectorized(3):
                    with T.block("T_layout_trans_1"):
                        T.where(3 <= i0_i1_i2_fused % 230 and i0_i1_i2_fused % 230 < 227)
                        ax0_1 = T.axis.spatial(8, i0_i1_i2_fused // 230 + ax0)
                        ax1_1 = T.axis.spatial(1, ax1)
                        ax2_1 = T.axis.spatial(224, i0_i1_i2_fused % 230 + -3 + ax2)
                        ax3_1, ax4 = T.axis.remap("SS", [ax3, ax4_fused])
                        T.reads(p0[ax0_1, ax1_1 * 3 + ax4, ax2_1, ax3_1])
                        T.writes(T_layout_trans_1[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                        T_layout_trans_1[ax0_1, ax1_1, ax2_1, ax3_1, ax4] = T.if_then_else(ax0_1 < 8 and ax1_1 * 3 + ax4 < 3 and ax2_1 < 224 and ax3_1 < 224, p1[ax0_1, ax1_1 * 3 + ax4, ax2_1, ax3_1], T.float32(0), dtype="float32")
            for i3 in T.serial(230):
                for i4_fused in T.vectorized(3):
                    with T.block("data_pad"):
                        i0_1 = T.axis.spatial(8, i0_i1_i2_fused // 230)
                        i1_1 = T.axis.spatial(1, 0)
                        i2_1 = T.axis.spatial(230, i0_i1_i2_fused % 230)
                        i3_1, i4_1 = T.axis.remap("SS", [i3, i4_fused])
                        T.reads(T_layout_trans_1[i0_1, i1_1, i2_1 - 3, i3_1 - 3, i4_1])
                        T.writes(data_pad[i0_1, i1_1, i2_1, i3_1, i4_1])
                        data_pad[i0_1, i1_1, i2_1, i3_1, i4_1] = T.if_then_else(3 <= i2_1 and i2_1 < 227 and 3 <= i3_1 and i3_1 < 227, T_layout_trans_1[i0_1, i1_1, i2_1 - 3, i3_1 - 3, i4_1], T.float32(0), dtype="float32")
        for i0_0_i1_0_i2_0_fused in T.parallel(896, annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(4, 1, 7, 7, 3):
                for ax5_fused in T.vectorized(4):
                    with T.block("T_layout_trans"):
                        ax0_2 = T.axis.spatial(16, i0_0_i1_0_i2_0_fused % 448 // 112 * 4 + ax0)
                        ax1_2, ax2_2, ax3_2, ax4_1, ax5 = T.axis.remap("SSSSS", [ax1, ax2, ax3, ax4, ax5_fused])
                        T.reads(p0[ax0_2 * 4 + ax5, ax1_2 * 3 + ax4_1, ax2_2, ax3_2])
                        T.writes(T_layout_trans[ax0_2, ax1_2, ax2_2, ax3_2, ax4_1, ax5])
                        T_layout_trans[ax0_2, ax1_2, ax2_2, ax3_2, ax4_1, ax5] = T.if_then_else(ax0_2 * 4 + ax5 < 64 and ax1_2 * 3 + ax4_1 < 3 and ax2_2 < 7 and ax3_2 < 7, p0[ax0_2 * 4 + ax5, ax1_2 * 3 + ax4_1, ax2_2, ax3_2], T.float32(0), dtype="float32")
            for i3_0, i4_0, i0_1, i1_1, i2_1, i3_1, i4_1 in T.grid(1, 1, 4, 1, 1, 56, 1):
                for i0_2_init, i1_2_init, i2_2_init, i3_2_init, i4_2_init, i0_3_init, i1_3_init, i2_3_init, i3_3_init in T.grid(1, 1, 1, 1, 1, 1, 4, 1, 2):
                    for i4_3_fused_init in T.vectorized(4):
                        with T.block("conv2d_NCHWc_init"):
                            n = T.axis.spatial(8, i0_2_init + i0_3_init + i0_0_i1_0_i2_0_fused // 448 * 4 + i0_1)
                            oc_chunk = T.axis.spatial(16, i0_0_i1_0_i2_0_fused % 448 // 112 * 4 + i1_1 * 4 + i1_2_init * 4 + i1_3_init)
                            oh = T.axis.spatial(112, i0_0_i1_0_i2_0_fused % 112 + i2_1 + i2_2_init + i2_3_init)
                            ow = T.axis.spatial(112, i3_0 * 112 + i3_1 * 2 + i3_2_init * 2 + i3_3_init)
                            oc_block = T.axis.spatial(4, i4_0 * 4 + i4_1 * 4 + i4_2_init * 4 + i4_3_fused_init)
                            T.reads()
                            T.writes(conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS", "workload":["conv2d_NCHWc.x86", ["TENSOR", [8, 1, 224, 224, 3], "float32"], ["TENSOR", [16, 1, 7, 7, 3, 4], "float32"], [2, 2], [3, 3, 3, 3], [1, 1], "NCHW3c", "NCHW4c", "float32"]})
                            conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block] = T.float32(0)
                for i5_0, i6_0, i7_0, i0_2, i1_2, i2_2, i3_2, i4_2, i5_1, i6_1, i7_1, i0_3, i1_3, i2_3, i3_3 in T.grid(3, 7, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 4, 1, 2):
                    for i4_3_fused in T.vectorized(4):
                        with T.block("conv2d_NCHWc_update"):
                            n = T.axis.spatial(8, i0_2 + i0_3 + i0_0_i1_0_i2_0_fused // 448 * 4 + i0_1)
                            oc_chunk = T.axis.spatial(16, i0_0_i1_0_i2_0_fused % 448 // 112 * 4 + i1_1 * 4 + i1_2 * 4 + i1_3)
                            oh = T.axis.spatial(112, i0_0_i1_0_i2_0_fused % 112 + i2_1 + i2_2 + i2_3)
                            ow = T.axis.spatial(112, i3_0 * 112 + i3_1 * 2 + i3_2 * 2 + i3_3)
                            oc_block = T.axis.spatial(4, i4_0 * 4 + i4_1 * 4 + i4_2 * 4 + i4_3_fused)
                            ic = T.axis.reduce(3, i5_1 + i5_0)
                            kh = T.axis.reduce(7, i6_0 + i6_1)
                            kw = T.axis.reduce(7, i7_0 * 7 + i7_1)
                            T.reads(conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block], data_pad[n, ic // 3, oh * 2 + kh, ow * 2 + kw, ic % 3], T_layout_trans[oc_chunk, ic // 3, kh, kw, ic % 3, oc_block])
                            T.writes(conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS", "workload":["conv2d_NCHWc.x86", ["TENSOR", [8, 1, 224, 224, 3], "float32"], ["TENSOR", [16, 1, 7, 7, 3, 4], "float32"], [2, 2], [3, 3, 3, 3], [1, 1], "NCHW3c", "NCHW4c", "float32"]})
                            conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block] + data_pad[n, ic // 3, oh * 2 + kh, ow * 2 + kw, ic % 3] * T_layout_trans[oc_chunk, ic // 3, kh, kw, ic % 3, oc_block]
                for ax0_3, ax1_3, ax2_3 in T.grid(1, 4, 1):
                    for ax3_ax4_fused in T.vectorized(8):
                        with T.block("conv2d_NCHWc_global"):
                            v0 = T.axis.spatial(8, i0_0_i1_0_i2_0_fused // 448 * 4 + i0_1 + ax0_3)
                            v1 = T.axis.spatial(16, i0_0_i1_0_i2_0_fused % 448 // 112 * 4 + ax1_3)
                            v2 = T.axis.spatial(112, i0_0_i1_0_i2_0_fused % 112 + ax2_3)
                            v3 = T.axis.spatial(112, i3_1 * 2 + ax3_ax4_fused // 4)
                            v4 = T.axis.spatial(4, ax3_ax4_fused % 4)
                            T.reads(conv2d_NCHWc_global[v0, v1, v2, v3, v4])
                            T.writes(conv2d_NCHWc[v0, v1, v2, v3, v4])
                            conv2d_NCHWc[v0, v1, v2, v3, v4] = conv2d_NCHWc_global[v0, v1, v2, v3, v4]
        for i0_i1_fused in T.parallel(512, annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
            for i2, i3 in T.grid(112, 112):
                with T.block("out_ptr0"):
                    ax0_4 = T.axis.spatial(8, i0_i1_fused // 64)
                    ax1_4 = T.axis.spatial(64, i0_i1_fused % 64)
                    ax2_4, ax3_3 = T.axis.remap("SS", [i2, i3])
                    T.reads(conv2d_NCHWc[ax0_4, ax1_4 // 4, ax2_4, ax3_3, ax1_4 % 4])
                    T.writes(out_ptr0[ax0_4, ax1_4, ax2_4, ax3_3])
                    out_ptr0[ax0_4, ax1_4, ax2_4, ax3_3] = T.if_then_else(ax0_4 < 8 and ax1_4 < 64 and ax2_4 < 112 and ax3_3 < 112, conv2d_NCHWc[ax0_4, ax1_4 // 4, ax2_4, ax3_3, ax1_4 % 4], T.float32(0), dtype="float32")
# fmt: on


class _conv:
    @staticmethod
    def convolution(
        x_shape,
        w_shape,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ):
        # Use transpose or normal
        use_transpose = transposed

        data = relay.var("data", shape=x_shape)
        weight = relay.var("weight", shape=w_shape)
        strides = tuple(stride)
        padding = tuple(padding)
        dilation = tuple(dilation)
        groups = int(groups)
        weight_shape = w_shape

        if use_transpose:
            channels = weight_shape[1] * groups
            in_channels = weight_shape[0]
        else:
            channels = weight_shape[0]
            in_channels = weight_shape[1]

        # Check if this is depth wise convolution
        # We need to reshape weight so that Relay could recognize this is depth wise
        # weight_shape[1] is always in_channels // groups
        # For depthwise, in_channels == groups, so weight_shape[1] == 1
        # If groups > 1 but weight_shape[1] != 1, this is group convolution
        if groups > 1 and in_channels == 1:
            channel_multiplier = channels // groups
            new_weight_shape = (groups, channel_multiplier) + tuple(weight_shape[2:])
            weight = _op.transform.reshape(weight, new_weight_shape)

        kernel_size = weight_shape[2:]
        use_bias = isinstance(bias, _expr.Expr)

        # We are trying to invoke various relay operations through a single conv_op variable.
        # However the function signatures for some operations have additional attributes so we
        # pass these in along with the standard ones.
        additional_arguments = dict()

        if use_transpose:
            if len(kernel_size) == 3:
                conv_op = _op.nn.conv3d_transpose
            elif len(kernel_size) == 2:
                conv_op = _op.nn.conv2d_transpose
            else:
                conv_op = _op.nn.conv1d_transpose
            output_padding = tuple(output_padding)
            additional_arguments["output_padding"] = output_padding

        else:
            if len(kernel_size) == 3:
                conv_op = _op.nn.conv3d
            elif len(kernel_size) == 2:
                conv_op = _op.nn.conv2d
            else:
                conv_op = _op.nn.conv1d

        if len(kernel_size) == 3:
            data_layout = "NCDHW"
            kernel_layout = "OIDHW"
        elif len(kernel_size) == 2:
            data_layout = "NCHW"
            kernel_layout = "OIHW"
            if use_transpose:
                # Transposed convolutions have IOHW layout.
                kernel_layout = "IOHW"
        else:
            data_layout = "NCW"
            kernel_layout = "OIW"

        # Conv1d does not currently support grouped convolution so we convert it to conv2d
        is_grouped_conv1d = False
        if groups > 1 and len(kernel_size) == 1 and not use_transpose:
            is_grouped_conv1d = True
            conv_op = _op.nn.conv2d
            kernel_size = [1] + kernel_size
            strides = (1,) + strides
            padding = (0,) + padding
            dilation = (1,) + dilation
            data = _op.expand_dims(data, axis=2)
            weight = _op.expand_dims(weight, axis=2)
            data_layout = "NCHW"
            kernel_layout = "OIHW"

        conv_out = conv_op(
            data,
            weight,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            channels=channels,
            kernel_size=kernel_size,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            out_layout="",
            out_dtype="",
            **additional_arguments,
        )
        if use_bias:
            res = _op.nn.bias_add(conv_out, bias)
        else:
            res = conv_out
        if is_grouped_conv1d:
            # Because we conducted grouped conv1d convolution through conv2d we must
            # squeeze the output to get the correct result.
            res = _op.squeeze(res, axis=[2])
        return res

    @staticmethod
    def _call(
        x,
        w,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ):
        relay_func = _conv.convolution(
            x.shape,
            w.shape,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
        mod = tvm.IRModule.from_expr(relay_func)
        target = tvm.target.Target("llvm --num-cores=16")
        lib = _conv.tune_with_tvm(mod, target, {})
        print(relay_func)
        ctx = tvm.cpu()
        m = graph_executor.GraphModule(lib["default"](ctx))
        m.set_input("data", tvm.nd.from_dlpack(x.detach().contiguous()))
        m.set_input("weight", tvm.nd.from_dlpack(w.detach().contiguous()))
        m.run()
        return torch.from_dlpack(m.get_output(0))

    @staticmethod
    def tune_with_tvm(mod, target, params):
        import tempfile

        tasks = ms.relay_integration.extract_tasks(mod, target, params)
        for tsk in tasks:
            print(tsk.dispatched[0].script())
        return _conv.tune_with_relay(mod, target, params)
        with tempfile.TemporaryDirectory() as work_dir:
            database = ms.relay_integration.tune_relay(
                mod=mod,
                target=tvm.target.Target("llvm --num-cores=12"),
                work_dir=work_dir,
                max_trials_global=20000,
                num_trials_per_iter=64,
                params=params,
                strategy="evolutionary",
            )
            lib = ms.relay_integration.compile_relay(
                database=database,
                mod=mod,
                target=tvm.target.Target("llvm --num-cores=12"),
                params=params,
            )
        return lib

    @staticmethod
    def tune_with_relay(mod, target, params):
        with tvm.transform.PassContext(opt_level=10):
            lib = relay.build(mod, target=target, params=params)
        return lib

    @staticmethod
    def forward(
        x,
        w,
        bias,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        transposed=False,
        output_padding=(0, 0),
        groups=1,
    ):
        return _conv._call(
            x,
            w,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )


conv = _conv.forward
