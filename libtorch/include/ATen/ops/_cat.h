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



#include <ATen/ops/_cat_ops.h>

namespace at {


// aten::_cat(Tensor[] tensors, int dim=0) -> Tensor
TORCH_API inline at::Tensor _cat(at::TensorList tensors, int64_t dim=0) {
    return at::_ops::_cat::call(tensors, dim);
}

// aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & _cat_out(at::Tensor & out, at::TensorList tensors, int64_t dim=0) {
    return at::_ops::_cat_out::call(tensors, dim, out);
}

// aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & _cat_outf(at::TensorList tensors, int64_t dim, at::Tensor & out) {
    return at::_ops::_cat_out::call(tensors, dim, out);
}

}
