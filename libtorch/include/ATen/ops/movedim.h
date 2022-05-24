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



#include <ATen/ops/movedim_ops.h>

namespace at {


// aten::movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
TORCH_API inline at::Tensor movedim(const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
    return at::_ops::movedim_intlist::call(self, source, destination);
}

// aten::movedim.int(Tensor(a) self, int source, int destination) -> Tensor(a)
TORCH_API inline at::Tensor movedim(const at::Tensor & self, int64_t source, int64_t destination) {
    return at::_ops::movedim_int::call(self, source, destination);
}

}
