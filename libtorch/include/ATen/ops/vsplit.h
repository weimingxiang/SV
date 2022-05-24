#pragma once

// @generated by tools/codegen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/vsplit_ops.h>

namespace at {


// aten::vsplit.int(Tensor(a -> *) self, int sections) -> Tensor(a)[]
TORCH_API inline ::std::vector<at::Tensor> vsplit(const at::Tensor & self, int64_t sections) {
    return at::_ops::vsplit_int::call(self, sections);
}

// aten::vsplit.array(Tensor(a -> *) self, int[] indices) -> Tensor(a)[]
TORCH_API inline ::std::vector<at::Tensor> vsplit(const at::Tensor & self, at::IntArrayRef indices) {
    return at::_ops::vsplit_array::call(self, indices);
}

}
