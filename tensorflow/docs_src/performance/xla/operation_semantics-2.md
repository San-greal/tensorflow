## CrossReplicaSum

See also
[`ComputationBuilder::CrossReplicaSum`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Computes a sum across replicas.

<b> `CrossReplicaSum(operand)` </b>

| Arguments    | Type                    | Semantics                          |
| ------------ | ----------------------- | ---------------------------------- |
| `operand`    | `ComputationDataHandle` | Array to sum across replicas.      |

The output shape is the same as the input shape. For example, if there are two
replicas and the operand has the value `(1.0, 2.5)` and `(3.0, 5.1)`
respectively on the two replicas, then the output value from this op will be
`(4.0, 7.6)` on both replicas.

Computing the result of CrossReplicaSum requires having one input from each
replica, so if one replica executes a CrossReplicaSum node more times than
another, then the former replica will wait forever. Since the replicas are all
running the same program, there are not a lot of ways for that to happen, but it
is possible when a while loop's condition depends on data from infeed and the
data that is infed causes the while loop to iterate more times on one replica
than another.

## CustomCall

See also
[`ComputationBuilder::CustomCall`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Call a user-provided function within a computation.

<b> `CustomCall(target_name, args..., shape)` </b>

| Arguments     | Type                     | Semantics                        |
| ------------- | ------------------------ | -------------------------------- |
| `target_name` | `string`                 | Name of the function. A call     |
:               :                          : instruction will be emitted      :
:               :                          : which targets this symbol name.  :
| `args`        | sequence of N            | N arguments of arbitrary type,   |
:               : `ComputationDataHandle`s : which will be passed to the      :
:               :                          : function.                        :
| `shape`       | `Shape`                  | Output shape of the function     |

The function signature is the same, regardless of the arity or type of args:

```
extern "C" void target_name(void* out, void** in);
```

For example, if CustomCall is used as follows:

```
let x = f32[2] {1,2};
let y = f32[2x3] {{10, 20, 30}, {40, 50, 60}};

CustomCall("myfunc", {x, y}, f32[3x3])
```

Here is an example of an implementation of `myfunc`:

```
extern "C" void myfunc(void* out, void** in) {
  float (&x)[2] = *static_cast<float(*)[2]>(in[0]);
  float (&y)[2][3] = *static_cast<float(*)[2][3]>(in[1]);
  EXPECT_EQ(1, x[0]);
  EXPECT_EQ(2, x[1]);
  EXPECT_EQ(10, y[0][0]);
  EXPECT_EQ(20, y[0][1]);
  EXPECT_EQ(30, y[0][2]);
  EXPECT_EQ(40, y[1][0]);
  EXPECT_EQ(50, y[1][1]);
  EXPECT_EQ(60, y[1][2]);
  float (&z)[3][3] = *static_cast<float(*)[3][3]>(out);
  z[0][0] = x[1] + y[1][0];
  // ...
}
```

The user-provided function must not have side-effects and its execution must be
idempotent.

> Note: The opaque nature of the user-provided function restricts optimization
> opportunities for the compiler. Try to express your computation in terms of
> native XLA ops whenever possible; only use CustomCall as a last resort.

## Dot

See also
[`ComputationBuilder::Dot`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

<b> `Dot(lhs, rhs)` </b>

Arguments | Type                    | Semantics
--------- | ----------------------- | ---------------
`lhs`     | `ComputationDataHandle` | array of type T
`rhs`     | `ComputationDataHandle` | array of type T

The exact semantics of this operation depend on the ranks of the operands:

| Input                   | Output                | Semantics               |
| ----------------------- | --------------------- | ----------------------- |
| vector [n] `dot` vector | scalar                | vector dot product      |
: [n]                     :                       :                         :
| matrix [m x k] `dot`    | vector [m]            | matrix-vector           |
: vector [k]              :                       : multiplication          :
| matrix [m x k] `dot`    | matrix [m x n]        | matrix-matrix           |
: matrix [k x n]          :                       : multiplication          :

The operation performs sum of products over the last dimension of `lhs` and the
one-before-last dimension of `rhs`. These are the "contracted" dimensions. The
contracted dimensions of `lhs` and `rhs` must be of the same size. In practice,
it can be used to perform dot products between vectors, vector/matrix
multiplications or matrix/matrix multiplications.

## Element-wise binary arithmetic operations

See also
[`ComputationBuilder::Add`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

A set of element-wise binary arithmetic operations is supported.

<b> `Op(lhs, rhs)` </b>

Where `Op` is one of `Add` (addition), `Sub` (subtraction), `Mul`
(multiplication), `Div` (division), `Rem` (remainder), `Max` (maximum), `Min`
(minimum), `LogicalAnd` (logical AND), or `LogicalOr` (logical OR).

Arguments | Type                    | Semantics
--------- | ----------------------- | ----------------------------------------
`lhs`     | `ComputationDataHandle` | left-hand-side operand: array of type T
`rhs`     | `ComputationDataHandle` | right-hand-side operand: array of type T

The arguments' shapes have to be either similar or compatible. See the
@{$broadcasting$broadcasting} documentation about what it means for shapes to
be compatible. The result of an operation has a shape which is the result of
broadcasting the two input arrays. In this variant, operations between arrays of
different ranks are *not* supported, unless one of the operands is a scalar.

When `Op` is `Rem`, the sign of the result is taken from the dividend, and the
absolute value of the result is always less than the divisor's absolute value.

An alternative variant with different-rank broadcasting support exists for these
operations:

<b> `Op(lhs, rhs, broadcast_dimensions)` </b>

Where `Op` is the same as above. This variant of the operation should be used
for arithmetic operations between arrays of different ranks (such as adding a
matrix to a vector).

The additional `broadcast_dimensions` operand is a slice of integers used to
expand the rank of the lower-rank operand up to the rank of the higher-rank
operand. `broadcast_dimensions` maps the dimensions of the lower-rank shape to
the dimensions of the higher-rank shape. The unmapped dimensions of the expanded
shape are filled with dimensions of size one. Degenerate-dimension broadcasting
then broadcasts the shapes along these degenerate dimension to equalize the
shapes of both operands. The semantics are described in detail on the
@{$broadcasting$broadcasting page}.

## Element-wise comparison operations

See also
[`ComputationBuilder::Eq`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

A set of standard element-wise binary comparison operations is supported. Note
that standard IEEE 754 floating-point comparison semantics apply when comparing
floating-point types.

<b> `Op(lhs, rhs)` </b>

Where `Op` is one of `Eq` (equal-to), `Ne` (not equal-to), `Ge`
(greater-or-equal-than), `Gt` (greater-than), `Le` (less-or-equal-than), `Le`
(less-than).

Arguments | Type                    | Semantics
--------- | ----------------------- | ----------------------------------------
`lhs`     | `ComputationDataHandle` | left-hand-side operand: array of type T
`rhs`     | `ComputationDataHandle` | right-hand-side operand: array of type T

The arguments' shapes have to be either similar or compatible. See the
@{$broadcasting$broadcasting} documentation about what it means for shapes to
be compatible. The result of an operation has a shape which is the result of
broadcasting the two input arrays with the element type `PRED`. In this variant,
operations between arrays of different ranks are *not* supported, unless one of
the operands is a scalar.

An alternative variant with different-rank broadcasting support exists for these
operations:

<b> `Op(lhs, rhs, broadcast_dimensions)` </b>

Where `Op` is the same as above. This variant of the operation should be used
for comparison operations between arrays of different ranks (such as adding a
matrix to a vector).

The additional `broadcast_dimensions` operand is a slice of integers specifying
the dimensions to use for broadcasting the operands. The semantics are described
in detail on the @{$broadcasting$broadcasting page}.

## Element-wise unary functions

ComputationBuilder supports these element-wise unary functions:

<b>`Abs(operand)`</b> Element-wise abs `x -> |x|`.

<b>`Ceil(operand)`</b> Element-wise ceil `x -> ⌈x⌉`.

<b>`Cos(operand)`</b> Element-wise cosine `x -> cos(x)`.

<b>`Exp(operand)`</b> Element-wise natural exponential `x -> e^x`.

<b>`Floor(operand)`</b> Element-wise floor `x -> ⌊x⌋`.

<b>`IsFinite(operand)`</b> Tests whether each element of `operand` is finite,
i.e., is not positive or negative infinity, and is not `NaN`. Returns an array
of `PRED` values with the same shape as the input, where each element is `true`
if and only if the corresponding input element is finite.

<b>`Log(operand)`</b> Element-wise natural logarithm `x -> ln(x)`.

<b>`LogicalNot(operand)`</b> Element-wise logical not `x -> !(x)`.

<b>`Neg(operand)`</b> Element-wise negation `x -> -x`.

<b>`Sign(operand)`</b> Element-wise sign operation `x -> sgn(x)` where

$$\text{sgn}(x) = \begin{cases} -1 & x < 0\\ 0 & x = 0\\ 1 & x > 0 \end{cases}$$

using the comparison operator of the element type of `operand`.

<b>`Tanh(operand)`</b> Element-wise hyperbolic tangent `x -> tanh(x)`.


Arguments | Type                    | Semantics
--------- | ----------------------- | ---------------------------
`operand` | `ComputationDataHandle` | The operand to the function

The function is applied to each element in the `operand` array, resulting in an
array with the same shape. It is allowed for `operand` to be a scalar (rank 0).


## BatchNormTraining

See also
[`ComputationBuilder::BatchNormTraining`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h) and
[`the original batch normalization paper`](https://arxiv.org/abs/1502.03167)
for a detailed description of the algorithm.

<b> Warning: Not implemented on GPU backend yet. </b>

Normalizes an array across batch and spatial dimensions.

<b> `BatchNormTraining(operand, scale, offset, epsilon, feature_index)` </b>

| Arguments       | Type                    | Semantics                        |
| --------------- | ----------------------- | -------------------------------- |
| `operand`       | `ComputationDataHandle` | n dimensional array to be        |
:                 :                         : normalized                       :
| `scale`         | `ComputationDataHandle` | 1 dimensional array              |
:                 :                         : (\\(\gamma\\))                   :
| `offset`        | `ComputationDataHandle` | 1 dimensional array              |
:                 :                         : (\\(\beta\\ )                    :
| `epsilon`       | `float`                 | Epsilon value (\\(\epsilon\\))   |
| `feature_index` | `int64`                 | Index to feature dimension       |
:                 :                         : in `operand`                     :


For each feature in the feature dimension (`feature_index` is the index for the
feature dimension in `operand`), the operation calculates the mean and variance
across all the other dimensions and use the mean and variance to normalize each
element in `operand`. If an invalid `feature_index` is passed, an error is
produced.

The algorithm goes as follows for each batch in `operand` \\(x\\) that
contains `m` elements with `w` and `h` as the size of spatial dimensions (
assuming `operand` is an 4 dimensional array):

- Calculates batch mean \\(\mu_l\\) for each feature `l` in feature dimension:
\\(\mu_l=\frac{1}{mwh}\sum_{i=1}^m\sum_{j=1}^w\sum_{k=1}^h x_{ijkl}\\)

- Calculates batch variance \\(\sigma^2_l\\):
\\(\sigma^2_l=\frac{1}{mwh}\sum_{i=1}^m\sum_{j=1}^w\sum_{k=1}^h (x_{ijkl} - \mu_l)^2\\)

- Normalizes, scales and shifts:
\\(y_{ijkl}=\frac{\gamma_l(x_{ijkl}-\mu_l)}{\sqrt[2]{\sigma^2_l+\epsilon}}+\beta_l\\)

The epsilon value, usually a small number, is added to avoid divide-by-zero errors.

The output type is a tuple of three ComputationDataHandles:

| Outputs      | Type                    | Semantics                            |
| ------------ | ----------------------- | -------------------------------------|
| `output`     | `ComputationDataHandle` | n dimensional array with the same    |
:              :                         : shape as input `operand` (y)         :
| `batch_mean` | `ComputationDataHandle` | 1 dimensional array (\\(\mu\\))      |
| `batch_var`  | `ComputationDataHandle` | 1 dimensional array (\\(\sigma^2\\)) |

The `batch_mean` and `batch_var` are moments calculated across the batch and
spatial dimensions using the formulars above.

## BatchNormInference

See also
[`ComputationBuilder::BatchNormInference`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

<b> Warning: Not implemented yet. </b>

Normalizes an array across batch and spatial dimensions.

<b> `BatchNormInference(operand, scale, offset, mean, variance, epsilon, feature_index)` </b>

| Arguments       | Type                    | Semantics                       |
| --------------  | ----------------------- | ------------------------------- |
| `operand`       | `ComputationDataHandle` | n dimensional array to be       |
:                 :                         : normalized                      :
| `scale`         | `ComputationDataHandle` | 1 dimensional array             |
| `offset`        | `ComputationDataHandle` | 1 dimensional array             |
| `mean`          | `ComputationDataHandle` | 1 dimensional array             |
| `variance`      | `ComputationDataHandle` | 1 dimensional array             |
| `epsilon`       | `float`                 | Epsilon value                   |
| `feature_index` | `int64`                 | Index to feature dimension in   |
:                 :                         : `operand`                       :

For each feature in the feature dimension (`feature_index` is the index for the
feature dimension in `operand`), the operation calculates the mean and variance
across all the other dimensions and use the mean and variance to normalize each
element in `operand`. If an invalid `feature_index` is passed, an error is
produced.

`BatchNormInference`  is equivalent to calling `BatchNormTraining` without
computing `mean` and `variance` for each batch. It uses the input `mean` and
`variance` instead as estimated values. The purpose of this op is to reduce
latency in inference, hence the name `BatchNormInference`.

The output is a n dimensional, normalized array with the same shape as input
`operand`.

## BatchNormGrad

See also
[`ComputationBuilder::BatchNormGrad`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

<b> Warning: Not implemented yet. </b>

Calculates gradients of batch norm.

<b> `BatchNormGrad(operand, scale, mean, variance, grad_output, epsilon, feature_index)` </b>

| Arguments       | Type                    | Semantics                        |
| --------------  | ----------------------- | -------------------------------- |
| `operand`       | `ComputationDataHandle` | n dimensional array to be        |
:                 :                         : normalized (x)                   :
| `scale`         | `ComputationDataHandle` | 1 dimensional array              |
:                 :                         : (\\(\gamma\\))                   :
| `mean`          | `ComputationDataHandle` | 1 dimensional array (\\(\mu\\))  |
| `variance`      | `ComputationDataHandle` | 1 dimensional array              |
:                 :                         : (\\(\sigma^2\\))                 :
| `grad_output`   | `ComputationDataHandle` | Gradients passed to              |
:                 :                         : `BatchNormTraining`              :
:                 :                         : (\\( \nabla y\\))                :
| `epsilon`       | `float`                 | Epsilon value (\\(\epsilon\\))   |
| `feature_index` | `int64`                 | Index to feature dimension in    |
:                 :                         : `operand`                        :

For each feature in the feature dimension (`feature_index` is the index for the
feature dimension in `operand`), the operation calculates the gradients with
respect to `operand`, `offset` and `scale` across all the other dimensions. If
an invalid `feature_index` is passed, an error is produced.

The three gradients are defined by the following formulas:

\\( \nabla x = \nabla y * \gamma * \sqrt{\sigma^2+\epsilon} \\)

\\( \nabla \gamma = sum(\nabla y * (x - \mu) * \sqrt{\sigma^2 + \epsilon}) \\)

\\( \nabla \beta = sum(\nabla y) \\)

The inputs `mean` and `variance` represents moments value
across batch and spatial dimensions.

The output type is a tuple of three ComputationDataHandles:

|Outputs       | Type                    | Semantics                           |
|------------- | ----------------------- | ------------------------------------|
|`grad_operand`| `ComputationDataHandle` | gradient with respect to input      |
:              :                         : `operand`                           :
|`grad_offset` | `ComputationDataHandle` | gradient with respect to input      |
:              :                         : `offset`                            :
|`grad_scale`  | `ComputationDataHandle` | gradient with respect to input      |
:              :                         : `scale`                             :


## GetTupleElement

See also
[`ComputationBuilder::GetTupleElement`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Indexes into a tuple with a compile-time-constant value.

The value must be a compile-time-constant so that shape inference can determine
the type of the resulting value.

This is analogous to `std::get<int N>(t)` in C++. Conceptually:

```
let v: f32[10] = f32[10]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
let s: s32 = 5;
let t: (f32[10], s32) = tuple(v, s);
let element_1: s32 = gettupleelement(t, 1);  // Inferred shape matches s32.
```

See also @{tf.tuple}.

## Infeed

See also
[`ComputationBuilder::Infeed`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

<b> `Infeed(shape)` </b>

| Argument | Type    | Semantics                                             |
| -------- | ------- | ----------------------------------------------------- |
| `shape`  | `Shape` | Shape of the data read from the Infeed interface. The |
:          :         : layout field of the shape must be set to match the    :
:          :         : layout of the data sent to the device; otherwise its  :
:          :         : behavior is undefined.                                :

Reads a single data item from the implicit Infeed streaming interface of the
device, interpreting the data as the given shape and its layout, and returns a
`ComputationDataHandle` of the data. Multiple Infeed operations are allowed in a
computation, but there must be a total order among the Infeed operations. For
example, two Infeeds in the code below have a total order since there is a
dependency between the while loops. The compiler issues an error if there isn't
a total order.

```
result1 = while (condition, init = init_value) {
  Infeed(shape)
}

result2 = while (condition, init = result1) {
  Infeed(shape)
}
```

Nested tuple shapes are not supported. For an empty tuple shape, the Infeed
operation is effectively a nop and proceeds without reading any data from the
Infeed of the device.

> Note: We plan to allow multiple Infeed operations without a total order, in
> which case the compiler will provide information about how the Infeed
> operations are serialized in the compiled program.

## Map

See also
[`ComputationBuilder::Map`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

<b> `Map(operands..., computation)` </b>

| Arguments         | Type                     | Semantics                     |
| ----------------- | ------------------------ | ----------------------------- |
| `operands`        | sequence of N            | N arrays of types T_0..T_{N-1}|
:                   : `ComputationDataHandle`s :                               :
| `computation`     | `Computation`            | computation of type `T_0,     |
:                   :                          : T_1, ..., T_{N + M -1} -> S`  :
:                   :                          : with N parameters of type T   :
:                   :                          : and M of arbitrary type       :
| `dimensions`       | `int64` array           | array of map dimensions    |
| `static_operands` | sequence of M            | M arrays of arbitrary type    |
:                   : `ComputationDataHandle`s :                               :

Applies a scalar function over the given `operands` arrays, producing an array
of the same dimensions where each element is the result of the mapped function
applied to the corresponding elements in the input arrays with `static_operands`
given as additional input to `computation`.

The mapped function is an arbitrary computation with the restriction that it has
N inputs of scalar type `T` and a single output with type `S`. The output has
the same dimensions as the operands except that the element type T is replaced
with S.

For example: `Map(op1, op2, op3, computation, par1)` maps `elem_out <-
computation(elem1, elem2, elem3, par1)` at each (multi-dimensional) index in the
input arrays to produce the output array.

## Pad

See also
[`ComputationBuilder::Pad`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

<b> `Pad(operand, padding_value, padding_config)` </b>

| Arguments        | Type                    | Semantics                     |
| ---------------- | ----------------------- | ----------------------------- |
| `operand`        | `ComputationDataHandle` | array of type `T`             |
| `padding_value`  | `ComputationDataHandle` | scalar of type `T` to fill in |
:                  :                         : the added padding             :
| `padding_config` | `PaddingConfig`         | padding amount on both edges  |
:                  :                         : (low, high) and between the   :
:                  :                         : elements of each dimension    :

Expands the given `operand` array by padding around the array as well as between
the elements of the array with the given `padding_value`. `padding_config`
specifies the amount of edge padding and the interior padding for each
dimension.

`PaddingConfig` is a repeated field of `PaddingConfigDimension`, which contains
three fields for each dimension: `edge_padding_low`, `edge_padding_high`, and
`interior_padding`. `edge_padding_low` and `edge_padding_high` specifies the
amount of padding added at the low-end (next to index 0) and the high-end (next
to the highest index) of each dimension respectively. The amount of edge padding
can be negative -- the absolute value of negative padding indicates the number
of elements to remove from the specified dimension. `interior_padding` specifies
the amount of padding added between any two elements in each dimension. Interior
padding occurs logically before edge padding, so in the case of negative edge
padding elements are removed from the interior-padded operand. This operation is
a no-op if the edge padding pairs are all (0, 0) and the interior padding values
are all 0. Figure below shows examples of different `edge_padding` and
`interior_padding` values for a two dimensional array.

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ops_pad.png">
</div>
