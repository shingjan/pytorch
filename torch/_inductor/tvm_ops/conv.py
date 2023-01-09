import uuid
import json
import os
import pathlib
from os.path import abspath

import tvm
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay import op as _op
from tvm.contrib import graph_executor
from tvm import meta_schedule as ms
from tvm.relay.frontend.pytorch import _convert_data_type


import torch


save_module = False
current_pt_dir = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
rt_libs_dir = f"{current_pt_dir}/rt_libs"
rt_modules_json_file = f"{rt_libs_dir}/modules_cuda.json"
ctx = tvm.cuda(0)
# ctx = tvm.cpu(0)
target = tvm.target.Target("nvidia/geforce-rtx-3070")
# target = tvm.target.Target("llvm --num-cores=16")


class conv_runtime_module:
    def __init__(
        self,
        x_shape,
        x_dtype,
        w_shape,
        w_dtype,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        uuid,
    ):
        self.x_shape = list(x_shape)
        self.x_dtype = str(x_dtype)
        self.w_shape = list(w_shape)
        self.w_dtype = str(w_dtype)
        # TODO(shingjan): bias could be tensor here
        # Need to be handled like x, w
        self.bias = bias
        self.stride = list(stride)
        self.padding = list(padding)
        self.dilation = list(dilation)
        self.transposed = bool(transposed)
        self.output_padding = list(output_padding)
        self.groups = int(groups)
        self.uuid = str(uuid)

    def is_same(self, other_module):
        if self.x_shape != other_module.x_shape:
            return False
        if self.x_dtype != other_module.x_dtype:
            return False
        if self.w_shape != other_module.w_shape:
            return False
        if self.w_dtype != other_module.w_dtype:
            return False
        if self.bias != other_module.bias:
            return False
        if self.stride != other_module.stride:
            return False
        if self.padding != other_module.padding:
            return False
        if self.dilation != other_module.dilation:
            return False
        if self.transposed != other_module.transposed:
            return False
        if self.output_padding != other_module.output_padding:
            return False
        if self.groups != other_module.groups:
            return False
        return True

    def set_runtime_module(self, runtime_module):
        self.runtime_module = runtime_module

    def get_runtime_module(self):
        if self.runtime_module == None:
            raise ValueError("runtime_module cannot be None")
        return self.runtime_module

    def set_runtime_lib(self, runtime_lib):
        self.runtime_lib = runtime_lib

    def get_runtime_lib(self):
        if self.runtime_lib == None:
            raise ValueError("runtime_lib cannot be None")
        return self.runtime_lib

    def update_json(self, filename):
        with open(filename, "r+") as file:
            file_data = json.load(file)
            file_data["modules"].append(json.dumps(self.__dict__))
            file.seek(0)
            json.dump(file_data, file, indent=4)


class _conv:
    def __init__(self):
        # list of conv_runtime_modules
        self.modules = []
        if save_module:
            with open(rt_modules_json_file, "r+") as file:
                file_data = json.load(file)
                for mod in file_data["modules"]:
                    module = json.loads(
                        mod, object_hook=lambda d: conv_runtime_module(**d)
                    )
                    lib = tvm.runtime.load_module(f"{rt_libs_dir}/{module.uuid}_lib.so")
                    m = graph_executor.GraphModule(lib["default"](ctx))
                    module.set_runtime_module(m)
                    self.modules.append(module)

    @staticmethod
    def convolution(
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
        # Use transpose or normal
        use_transpose = transposed

        data = relay.var(
            "data",
            shape=x.shape,
            dtype=_convert_data_type(str(x.dtype), "float32"),
        )
        weight = relay.var(
            "weight",
            shape=w.shape,
            dtype=_convert_data_type(str(w.dtype), "float32"),
        )
        strides = tuple(stride)
        padding = tuple(padding)
        dilation = tuple(dilation)
        groups = int(groups)
        weight_shape = w.shape

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
        use_bias = bias != None

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
            bias = relay.var(
                "bias",
                shape=bias.shape,
                dtype=_convert_data_type(str(bias.dtype), "float32"),
            )
            res = _op.nn.bias_add(conv_out, bias)
        else:
            res = conv_out
        if is_grouped_conv1d:
            # Because we conducted grouped conv1d convolution through conv2d we must
            # squeeze the output to get the correct result.
            res = _op.squeeze(res, axis=[2])
        return res

    def exist_module(self, module):
        for mod in self.modules:
            if mod.is_same(module):
                module.uuid = mod.uuid
                module.set_runtime_module(mod.get_runtime_module())
                return True
        return False

    def _call(
        self,
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
        module = conv_runtime_module(
            x.shape,
            x.dtype,
            w.shape,
            w.dtype,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            uuid.uuid4(),
        )
        if self.exist_module(module) == False:
            relay_func = _conv.convolution(
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
            mod = tvm.IRModule.from_expr(relay_func)
            lib = _conv.tune_with_tvm(mod, target, {})
            if save_module:
                lib.export_library(f"{rt_libs_dir}/{module.uuid}_lib.so")
                module.update_json(filename=rt_modules_json_file)
            m = graph_executor.GraphModule(lib["default"](ctx))
            module.set_runtime_module(m)
            # put it back into the cache
            self.modules.append(module)

        m = module.get_runtime_module()
        m.set_input("data", tvm.nd.from_dlpack(x.detach().contiguous()))
        m.set_input("weight", tvm.nd.from_dlpack(w.detach().contiguous()))
        m.run()
        res = m.get_output(0)
        output_tensor = tvm.nd.empty(res.shape, res.dtype, ctx).copyfrom(res)
        return torch.from_dlpack(output_tensor)

    @staticmethod
    def tune_with_tvm(mod, target, params):
        return _conv.tune_with_relay(mod, target, params)
        import tempfile

        with tempfile.TemporaryDirectory() as work_dir:
            database = ms.relay_integration.tune_relay(
                mod=mod,
                target=target,
                work_dir=work_dir,
                max_trials_global=20000,
                num_trials_per_iter=64,
                params=params,
                strategy="evolutionary",
            )
            lib = ms.relay_integration.compile_relay(
                database=database,
                mod=mod,
                target=target,
                params=params,
            )
        return lib

    @staticmethod
    def tune_with_relay(mod, target, params):
        with tvm.transform.PassContext(opt_level=10):
            lib = relay.build(mod, target=target, params=params)
        return lib

    def forward(
        self,
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
        return self._call(
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


_conv_object = _conv()

conv = _conv_object.forward
