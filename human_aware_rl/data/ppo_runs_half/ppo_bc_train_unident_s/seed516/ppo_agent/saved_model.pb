��
�5�5
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
Q
CheckNumerics
tensor"T
output"T"
Ttype:
2"
messagestring
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
n
LeakyReluGrad
	gradients"T
features"T
	backprops"T"
alphafloat%��L>"
Ttype0:
2
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
;
Minimum
x"T
y"T
z"T"
Ttype:

2	�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
�
StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.13.12b'v1.13.0-rc2-5-g6612da8'б
x
ppo_agent/ppo2_model/ObPlaceholder*
dtype0*&
_output_shapes
:	*
shape:	
�
Lppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
dtype0
�
Jppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/minConst*
valueB
 *����*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
dtype0*
_output_shapes
: 
�
Jppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *���=*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel
�
Tppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/RandomUniformRandomUniformLppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
seed2*
dtype0*&
_output_shapes
:*
seed�
�
Jppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/subSubJppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/maxJppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
_output_shapes
: 
�
Jppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/mulMulTppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/RandomUniformJppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
Fppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniformAddJppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/mulJppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
+ppo_agent/ppo2_model/pi/conv_initial/kernel
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
	container *
shape:
�
2ppo_agent/ppo2_model/pi/conv_initial/kernel/AssignAssign+ppo_agent/ppo2_model/pi/conv_initial/kernelFppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
0ppo_agent/ppo2_model/pi/conv_initial/kernel/readIdentity+ppo_agent/ppo2_model/pi/conv_initial/kernel*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
;ppo_agent/ppo2_model/pi/conv_initial/bias/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
dtype0*
_output_shapes
:
�
)ppo_agent/ppo2_model/pi/conv_initial/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
	container 
�
0ppo_agent/ppo2_model/pi/conv_initial/bias/AssignAssign)ppo_agent/ppo2_model/pi/conv_initial/bias;ppo_agent/ppo2_model/pi/conv_initial/bias/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
.ppo_agent/ppo2_model/pi/conv_initial/bias/readIdentity)ppo_agent/ppo2_model/pi/conv_initial/bias*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
2ppo_agent/ppo2_model/pi/conv_initial/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
�
+ppo_agent/ppo2_model/pi/conv_initial/Conv2DConv2Dppo_agent/ppo2_model/Ob0ppo_agent/ppo2_model/pi/conv_initial/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:	
�
,ppo_agent/ppo2_model/pi/conv_initial/BiasAddBiasAdd+ppo_agent/ppo2_model/pi/conv_initial/Conv2D.ppo_agent/ppo2_model/pi/conv_initial/bias/read*
T0*
data_formatNHWC*&
_output_shapes
:	
�
.ppo_agent/ppo2_model/pi/conv_initial/LeakyRelu	LeakyRelu,ppo_agent/ppo2_model/pi/conv_initial/BiasAdd*
T0*
alpha%��L>*&
_output_shapes
:	
�
Fppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
dtype0*
_output_shapes
:
�
Dppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/minConst*
valueB
 *�{�*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
dtype0*
_output_shapes
: 
�
Dppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *�{�=*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
dtype0
�
Nppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformFppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/shape*
seed2*
dtype0*&
_output_shapes
:*
seed�*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel
�
Dppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/subSubDppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/maxDppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/min*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
_output_shapes
: 
�
Dppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/mulMulNppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/RandomUniformDppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/sub*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
@ppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniformAddDppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/mulDppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel
�
%ppo_agent/ppo2_model/pi/conv_0/kernel
VariableV2*
shared_name *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:
�
,ppo_agent/ppo2_model/pi/conv_0/kernel/AssignAssign%ppo_agent/ppo2_model/pi/conv_0/kernel@ppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_0/kernel/readIdentity%ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel
�
5ppo_agent/ppo2_model/pi/conv_0/bias/Initializer/zerosConst*
valueB*    *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
dtype0*
_output_shapes
:
�
#ppo_agent/ppo2_model/pi/conv_0/bias
VariableV2*
shared_name *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_0/bias/AssignAssign#ppo_agent/ppo2_model/pi/conv_0/bias5ppo_agent/ppo2_model/pi/conv_0/bias/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
(ppo_agent/ppo2_model/pi/conv_0/bias/readIdentity#ppo_agent/ppo2_model/pi/conv_0/bias*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
}
,ppo_agent/ppo2_model/pi/conv_0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%ppo_agent/ppo2_model/pi/conv_0/Conv2DConv2D.ppo_agent/ppo2_model/pi/conv_initial/LeakyRelu*ppo_agent/ppo2_model/pi/conv_0/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:	
�
&ppo_agent/ppo2_model/pi/conv_0/BiasAddBiasAdd%ppo_agent/ppo2_model/pi/conv_0/Conv2D(ppo_agent/ppo2_model/pi/conv_0/bias/read*
T0*
data_formatNHWC*&
_output_shapes
:	
�
(ppo_agent/ppo2_model/pi/conv_0/LeakyRelu	LeakyRelu&ppo_agent/ppo2_model/pi/conv_0/BiasAdd*
T0*
alpha%��L>*&
_output_shapes
:	
�
Fppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
dtype0*
_output_shapes
:
�
Dppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/minConst*
valueB
 *�{�*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
dtype0*
_output_shapes
: 
�
Dppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *�{�=*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
dtype0*
_output_shapes
: 
�
Nppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformFppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*
seed�*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
seed2(
�
Dppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/subSubDppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/maxDppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/min*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
_output_shapes
: 
�
Dppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/mulMulNppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/RandomUniformDppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/sub*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
@ppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniformAddDppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/mulDppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/min*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:*
T0
�
%ppo_agent/ppo2_model/pi/conv_1/kernel
VariableV2*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel
�
,ppo_agent/ppo2_model/pi/conv_1/kernel/AssignAssign%ppo_agent/ppo2_model/pi/conv_1/kernel@ppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
�
*ppo_agent/ppo2_model/pi/conv_1/kernel/readIdentity%ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel
�
5ppo_agent/ppo2_model/pi/conv_1/bias/Initializer/zerosConst*
valueB*    *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
dtype0*
_output_shapes
:
�
#ppo_agent/ppo2_model/pi/conv_1/bias
VariableV2*
shared_name *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_1/bias/AssignAssign#ppo_agent/ppo2_model/pi/conv_1/bias5ppo_agent/ppo2_model/pi/conv_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias
�
(ppo_agent/ppo2_model/pi/conv_1/bias/readIdentity#ppo_agent/ppo2_model/pi/conv_1/bias*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
}
,ppo_agent/ppo2_model/pi/conv_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%ppo_agent/ppo2_model/pi/conv_1/Conv2DConv2D(ppo_agent/ppo2_model/pi/conv_0/LeakyRelu*ppo_agent/ppo2_model/pi/conv_1/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
&ppo_agent/ppo2_model/pi/conv_1/BiasAddBiasAdd%ppo_agent/ppo2_model/pi/conv_1/Conv2D(ppo_agent/ppo2_model/pi/conv_1/bias/read*
T0*
data_formatNHWC*&
_output_shapes
:
�
(ppo_agent/ppo2_model/pi/conv_1/LeakyRelu	LeakyRelu&ppo_agent/ppo2_model/pi/conv_1/BiasAdd*
T0*
alpha%��L>*&
_output_shapes
:
~
-ppo_agent/ppo2_model/pi/flatten/Reshape/shapeConst*
_output_shapes
:*
valueB"   ����*
dtype0
�
'ppo_agent/ppo2_model/pi/flatten/ReshapeReshape(ppo_agent/ppo2_model/pi/conv_1/LeakyRelu-ppo_agent/ppo2_model/pi/flatten/Reshape/shape*
_output_shapes
:	�*
T0*
Tshape0
�
Eppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"  @   *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
dtype0*
_output_shapes
:
�
Cppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *�ν*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
dtype0
�
Cppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
dtype0*
_output_shapes
: 
�
Mppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniformEppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/shape*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
seed2<*
dtype0*
_output_shapes
:	�@*
seed�
�
Cppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/subSubCppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/maxCppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel
�
Cppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/mulMulMppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/RandomUniformCppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/sub*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
?ppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniformAddCppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/mulCppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
$ppo_agent/ppo2_model/pi/dense/kernel
VariableV2*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
	container *
shape:	�@*
dtype0*
_output_shapes
:	�@
�
+ppo_agent/ppo2_model/pi/dense/kernel/AssignAssign$ppo_agent/ppo2_model/pi/dense/kernel?ppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
)ppo_agent/ppo2_model/pi/dense/kernel/readIdentity$ppo_agent/ppo2_model/pi/dense/kernel*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
4ppo_agent/ppo2_model/pi/dense/bias/Initializer/zerosConst*
valueB@*    *5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
dtype0*
_output_shapes
:@
�
"ppo_agent/ppo2_model/pi/dense/bias
VariableV2*
shared_name *5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
)ppo_agent/ppo2_model/pi/dense/bias/AssignAssign"ppo_agent/ppo2_model/pi/dense/bias4ppo_agent/ppo2_model/pi/dense/bias/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(
�
'ppo_agent/ppo2_model/pi/dense/bias/readIdentity"ppo_agent/ppo2_model/pi/dense/bias*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
$ppo_agent/ppo2_model/pi/dense/MatMulMatMul'ppo_agent/ppo2_model/pi/flatten/Reshape)ppo_agent/ppo2_model/pi/dense/kernel/read*
_output_shapes

:@*
transpose_a( *
transpose_b( *
T0
�
%ppo_agent/ppo2_model/pi/dense/BiasAddBiasAdd$ppo_agent/ppo2_model/pi/dense/MatMul'ppo_agent/ppo2_model/pi/dense/bias/read*
T0*
data_formatNHWC*
_output_shapes

:@
�
'ppo_agent/ppo2_model/pi/dense/LeakyRelu	LeakyRelu%ppo_agent/ppo2_model/pi/dense/BiasAdd*
_output_shapes

:@*
T0*
alpha%��L>
�
Gppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
dtype0*
_output_shapes
:
�
Eppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *׳]�*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
dtype0*
_output_shapes
: 
�
Eppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳]>*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
dtype0*
_output_shapes
: 
�
Oppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformGppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*
seed�*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
seed2M
�
Eppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/subSubEppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/maxEppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes
: 
�
Eppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/mulMulOppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/RandomUniformEppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/sub*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@*
T0
�
Appo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniformAddEppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/mulEppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/min*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@*
T0
�
&ppo_agent/ppo2_model/pi/dense_1/kernel
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
	container *
shape
:@@
�
-ppo_agent/ppo2_model/pi/dense_1/kernel/AssignAssign&ppo_agent/ppo2_model/pi/dense_1/kernelAppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
+ppo_agent/ppo2_model/pi/dense_1/kernel/readIdentity&ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel
�
6ppo_agent/ppo2_model/pi/dense_1/bias/Initializer/zerosConst*
valueB@*    *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
dtype0*
_output_shapes
:@
�
$ppo_agent/ppo2_model/pi/dense_1/bias
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
	container 
�
+ppo_agent/ppo2_model/pi/dense_1/bias/AssignAssign$ppo_agent/ppo2_model/pi/dense_1/bias6ppo_agent/ppo2_model/pi/dense_1/bias/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
)ppo_agent/ppo2_model/pi/dense_1/bias/readIdentity$ppo_agent/ppo2_model/pi/dense_1/bias*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@*
T0
�
&ppo_agent/ppo2_model/pi/dense_1/MatMulMatMul'ppo_agent/ppo2_model/pi/dense/LeakyRelu+ppo_agent/ppo2_model/pi/dense_1/kernel/read*
T0*
_output_shapes

:@*
transpose_a( *
transpose_b( 
�
'ppo_agent/ppo2_model/pi/dense_1/BiasAddBiasAdd&ppo_agent/ppo2_model/pi/dense_1/MatMul)ppo_agent/ppo2_model/pi/dense_1/bias/read*
T0*
data_formatNHWC*
_output_shapes

:@
�
)ppo_agent/ppo2_model/pi/dense_1/LeakyRelu	LeakyRelu'ppo_agent/ppo2_model/pi/dense_1/BiasAdd*
T0*
alpha%��L>*
_output_shapes

:@
�
Gppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
dtype0*
_output_shapes
:
�
Eppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *׳]�*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
dtype0
�
Eppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳]>*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
dtype0*
_output_shapes
: 
�
Oppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniformGppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*
seed�*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
seed2^
�
Eppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/subSubEppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/maxEppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes
: 
�
Eppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/mulMulOppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformEppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
Appo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniformAddEppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/mulEppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
&ppo_agent/ppo2_model/pi/dense_2/kernel
VariableV2*
shared_name *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
�
-ppo_agent/ppo2_model/pi/dense_2/kernel/AssignAssign&ppo_agent/ppo2_model/pi/dense_2/kernelAppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
+ppo_agent/ppo2_model/pi/dense_2/kernel/readIdentity&ppo_agent/ppo2_model/pi/dense_2/kernel*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
6ppo_agent/ppo2_model/pi/dense_2/bias/Initializer/zerosConst*
valueB@*    *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
dtype0*
_output_shapes
:@
�
$ppo_agent/ppo2_model/pi/dense_2/bias
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
	container 
�
+ppo_agent/ppo2_model/pi/dense_2/bias/AssignAssign$ppo_agent/ppo2_model/pi/dense_2/bias6ppo_agent/ppo2_model/pi/dense_2/bias/Initializer/zeros*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
)ppo_agent/ppo2_model/pi/dense_2/bias/readIdentity$ppo_agent/ppo2_model/pi/dense_2/bias*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
&ppo_agent/ppo2_model/pi/dense_2/MatMulMatMul)ppo_agent/ppo2_model/pi/dense_1/LeakyRelu+ppo_agent/ppo2_model/pi/dense_2/kernel/read*
_output_shapes

:@*
transpose_a( *
transpose_b( *
T0
�
'ppo_agent/ppo2_model/pi/dense_2/BiasAddBiasAdd&ppo_agent/ppo2_model/pi/dense_2/MatMul)ppo_agent/ppo2_model/pi/dense_2/bias/read*
_output_shapes

:@*
T0*
data_formatNHWC
�
)ppo_agent/ppo2_model/pi/dense_2/LeakyRelu	LeakyRelu'ppo_agent/ppo2_model/pi/dense_2/BiasAdd*
T0*
alpha%��L>*
_output_shapes

:@
]
ppo_agent/ppo2_model/ConstConst*
_output_shapes
: *
valueB *
dtype0
{
*ppo_agent/ppo2_model/flatten/Reshape/shapeConst*
_output_shapes
:*
valueB"   ����*
dtype0
�
$ppo_agent/ppo2_model/flatten/ReshapeReshape)ppo_agent/ppo2_model/pi/dense_2/LeakyRelu*ppo_agent/ppo2_model/flatten/Reshape/shape*
T0*
Tshape0*
_output_shapes

:@
}
,ppo_agent/ppo2_model/flatten_1/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"   ����
�
&ppo_agent/ppo2_model/flatten_1/ReshapeReshape)ppo_agent/ppo2_model/pi/dense_2/LeakyRelu,ppo_agent/ppo2_model/flatten_1/Reshape/shape*
T0*
Tshape0*
_output_shapes

:@
�
3ppo_agent/ppo2_model/pi/w/Initializer/initial_valueConst*�
value�B�@"��p9�>-:��X: K��,�u�;�2�8�@I�4�ƺ�Ϫ9��w:*쏺̡�:��ǺC��*�$�O7�>[;J.
:k��:�Z��
9X�9,�;��18���*��p9+��u:q�� .:����I�"�̸䗏8&wh�h��:b0�?3G:x�9�"�:K�0��jV�����ӺL&�:Hu����4����R�+:��<�7�;�5�Û�H�f�kH;��*:[vϹ��:t���1�9��:լf�y���Ҵ9n��k�n:_V���:=��9�:�d��LuںmȰ�+@�:���9vӺn�8+~y:0��:!{���r�*k[:���7p�7��M:ҭ���_�}b��氺}O�9l�:E(��:B��:�W��r8�:c�5��*�
�M:��{�W�S�mߺ^k;^�39脺�#�:�35��2:j�:�
��:ݗc:?��8\�U��o�2��,:����;x���ʧ���Wp�2I5�<&�9+^:e��9��꺸ᢺ� :6n"���"�'fйu�?������:�m�A�\9�rѹX�I9#��9y
�8�� �:D1j;g�:�9K㹵�:]�:���ӕ9�J7�Iw��9�:_�G��;���:2�\6�:�tT:y�:�c:�����vr:u45��!��n�:fv:��:����:��%��9󤣺^:�!�^�3�_¸�4S:[�/��猺��^8v��*��:���:�;�29�Z��M��:��	;���9���9'���!�����:�>��z�;bn:�P�:9�?��o�����ˏ��@�x�9�`!��P��69):ۮ�Ox�Ãƺ!�A��HQ:����:_�H���9�kѸ���m�M;��-:Ҹ��S��e�Z <�e��:�.��6�:����
24���2;F��:6:*�2:���:b��:��:���G���j�:�X�8��:�pX:FL9�@J�S��S5;";;D��8��:�˅8��=�f��>w:�Ȉ:}��Y��c�C�-�9^չ Vk� Ӻ��:@r�$�8*�.���S���ٺ��)��?5��*�[m�:�œ�y;!����0�:JR:y�j:���:P�K��?� #=�c?Ź��9��#:zN��Z�����;gU�i��P�:<��:ތx��^�:��9�I:��5�E������:��:��9u�=:��ݺš:�Ǒ��b���.�o�e����:���:)3:S��9��0���x:���W0<��m`8�&9A�:��9��(;���:3w�:�J:�
a�lZ�����#��:4K�f�;��d��r(9"A.;rG�:!�:frg:���M:#��6�v����`�8�_�:f,T9V����;Ӛ:μ:�c�:d�:�"G����]!�8d M�r$��$oV��e�9�+9c"�:VI���׳��	9�����]�|9�J;	:�Y-:�6:��ܹ��ܺ���b�:�A�9Sk�:R(����;v�3���	;ϝ���2�9x�:9*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
dtype0*
_output_shapes

:@
�
ppo_agent/ppo2_model/pi/w
VariableV2*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
�
 ppo_agent/ppo2_model/pi/w/AssignAssignppo_agent/ppo2_model/pi/w3ppo_agent/ppo2_model/pi/w/Initializer/initial_value*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
ppo_agent/ppo2_model/pi/w/readIdentityppo_agent/ppo2_model/pi/w*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
+ppo_agent/ppo2_model/pi/b/Initializer/ConstConst*
valueB*    *,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
dtype0*
_output_shapes
:
�
ppo_agent/ppo2_model/pi/b
VariableV2*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
 ppo_agent/ppo2_model/pi/b/AssignAssignppo_agent/ppo2_model/pi/b+ppo_agent/ppo2_model/pi/b/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
ppo_agent/ppo2_model/pi/b/readIdentityppo_agent/ppo2_model/pi/b*
_output_shapes
:*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b
�
 ppo_agent/ppo2_model/pi_1/MatMulMatMul&ppo_agent/ppo2_model/flatten_1/Reshapeppo_agent/ppo2_model/pi/w/read*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
�
ppo_agent/ppo2_model/pi_1/addAdd ppo_agent/ppo2_model/pi_1/MatMulppo_agent/ppo2_model/pi/b/read*
T0*
_output_shapes

:
o
ppo_agent/ppo2_model/SoftmaxSoftmaxppo_agent/ppo2_model/pi_1/add*
T0*
_output_shapes

:
t
!ppo_agent/ppo2_model/action_probsIdentityppo_agent/ppo2_model/Softmax*
T0*
_output_shapes

:
k
ppo_agent/ppo2_model/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
l
'ppo_agent/ppo2_model/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0
l
'ppo_agent/ppo2_model/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
1ppo_agent/ppo2_model/random_uniform/RandomUniformRandomUniformppo_agent/ppo2_model/Shape*
T0*
dtype0*
_output_shapes

:*
seed2�*
seed�
�
'ppo_agent/ppo2_model/random_uniform/subSub'ppo_agent/ppo2_model/random_uniform/max'ppo_agent/ppo2_model/random_uniform/min*
T0*
_output_shapes
: 
�
'ppo_agent/ppo2_model/random_uniform/mulMul1ppo_agent/ppo2_model/random_uniform/RandomUniform'ppo_agent/ppo2_model/random_uniform/sub*
T0*
_output_shapes

:
�
#ppo_agent/ppo2_model/random_uniformAdd'ppo_agent/ppo2_model/random_uniform/mul'ppo_agent/ppo2_model/random_uniform/min*
_output_shapes

:*
T0
m
ppo_agent/ppo2_model/LogLog#ppo_agent/ppo2_model/random_uniform*
T0*
_output_shapes

:
b
ppo_agent/ppo2_model/NegNegppo_agent/ppo2_model/Log*
_output_shapes

:*
T0
d
ppo_agent/ppo2_model/Log_1Logppo_agent/ppo2_model/Neg*
_output_shapes

:*
T0
�
ppo_agent/ppo2_model/subSubppo_agent/ppo2_model/pi_1/addppo_agent/ppo2_model/Log_1*
T0*
_output_shapes

:
p
%ppo_agent/ppo2_model/ArgMax/dimensionConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
ppo_agent/ppo2_model/ArgMaxArgMaxppo_agent/ppo2_model/sub%ppo_agent/ppo2_model/ArgMax/dimension*
T0*
output_type0	*
_output_shapes
:*

Tidx0
i
ppo_agent/ppo2_model/actionIdentityppo_agent/ppo2_model/ArgMax*
_output_shapes
:*
T0	
j
%ppo_agent/ppo2_model/one_hot/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
&ppo_agent/ppo2_model/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
"ppo_agent/ppo2_model/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
�
ppo_agent/ppo2_model/one_hotOneHotppo_agent/ppo2_model/action"ppo_agent/ppo2_model/one_hot/depth%ppo_agent/ppo2_model/one_hot/on_value&ppo_agent/ppo2_model/one_hot/off_value*
T0*
axis���������*
TI0	*
_output_shapes

:
}
;ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:

=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
~
<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
:ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/SubSub=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Rank_1<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
�
Bppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice/beginPack:ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N*
_output_shapes
:
�
Appo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/SliceSlice>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Shape_1Bppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice/beginAppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:
�
Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
�
Bppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concatConcatV2Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat/values_0<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/SliceBppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/ReshapeReshapeppo_agent/ppo2_model/pi_1/add=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*
_output_shapes

:

=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Shape_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_1Sub=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Rank_2>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
�
Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_1/beginPack<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_1*
_output_shapes
:*
T0*

axis *
N
�
Cppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_1Slice>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Shape_2Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_1/beginCppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
Hppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat_1/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
�
Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat_1ConcatV2Hppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat_1/values_0>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_1Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Reshape_1Reshapeppo_agent/ppo2_model/one_hot?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat_1*
_output_shapes

:*
T0*
Tshape0
�
6ppo_agent/ppo2_model/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Reshape@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Reshape_1*
T0*$
_output_shapes
::
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
�
<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_2Sub;ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Rank>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_2/y*
_output_shapes
: *
T0
�
Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
Cppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_2/sizePack<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_2Slice<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/ShapeDppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_2/beginCppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_2/size*
_output_shapes
:*
Index0*
T0
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Reshape_2Reshape6ppo_agent/ppo2_model/softmax_cross_entropy_with_logits>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0*
_output_shapes
:
�
3ppo_agent/ppo2_model/vf/w/Initializer/initial_valueConst*�
value�B�@"��˪� E�à>�y��!	2�������^&=~�n�ķ��2� ���j"���ս-��=��5�>��Z���n�Rl�F�G��54>�Գ�S�}=-pC=sb2�u>��@�?�>�>>�墼F�:��W>��#�r� 90�.>k��>�C�w
K=��Ľ�1绕�}���>$��=n����.<Zv=���ݞ��n�=��>��,>��	=��M=���P��`�=y��=�2����<-���V8��q���Ƚ*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
dtype0*
_output_shapes

:@
�
ppo_agent/ppo2_model/vf/w
VariableV2*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
 ppo_agent/ppo2_model/vf/w/AssignAssignppo_agent/ppo2_model/vf/w3ppo_agent/ppo2_model/vf/w/Initializer/initial_value*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
ppo_agent/ppo2_model/vf/w/readIdentityppo_agent/ppo2_model/vf/w*
_output_shapes

:@*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w
�
+ppo_agent/ppo2_model/vf/b/Initializer/ConstConst*
valueB*    *,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
dtype0*
_output_shapes
:
�
ppo_agent/ppo2_model/vf/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
	container *
shape:
�
 ppo_agent/ppo2_model/vf/b/AssignAssignppo_agent/ppo2_model/vf/b+ppo_agent/ppo2_model/vf/b/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
ppo_agent/ppo2_model/vf/b/readIdentityppo_agent/ppo2_model/vf/b*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
ppo_agent/ppo2_model/vf/MatMulMatMul$ppo_agent/ppo2_model/flatten/Reshapeppo_agent/ppo2_model/vf/w/read*
_output_shapes

:*
transpose_a( *
transpose_b( *
T0
�
ppo_agent/ppo2_model/vf/addAddppo_agent/ppo2_model/vf/MatMulppo_agent/ppo2_model/vf/b/read*
T0*
_output_shapes

:
y
(ppo_agent/ppo2_model/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
{
*ppo_agent/ppo2_model/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
{
*ppo_agent/ppo2_model/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
"ppo_agent/ppo2_model/strided_sliceStridedSliceppo_agent/ppo2_model/vf/add(ppo_agent/ppo2_model/strided_slice/stack*ppo_agent/ppo2_model/strided_slice/stack_1*ppo_agent/ppo2_model/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
T0*
Index0
o
ppo_agent/ppo2_model/valueIdentity"ppo_agent/ppo2_model/strided_slice*
T0*
_output_shapes
:
|
ppo_agent/ppo2_model/Ob_1Placeholder*
dtype0*'
_output_shapes
:�	*
shape:�	
�
4ppo_agent/ppo2_model/pi_2/conv_initial/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
-ppo_agent/ppo2_model/pi_2/conv_initial/Conv2DConv2Dppo_agent/ppo2_model/Ob_10ppo_agent/ppo2_model/pi/conv_initial/kernel/read*
paddingSAME*'
_output_shapes
:�	*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
.ppo_agent/ppo2_model/pi_2/conv_initial/BiasAddBiasAdd-ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D.ppo_agent/ppo2_model/pi/conv_initial/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:�	
�
0ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu	LeakyRelu.ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd*
alpha%��L>*'
_output_shapes
:�	*
T0

.ppo_agent/ppo2_model/pi_2/conv_0/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
'ppo_agent/ppo2_model/pi_2/conv_0/Conv2DConv2D0ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu*ppo_agent/ppo2_model/pi/conv_0/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�	
�
(ppo_agent/ppo2_model/pi_2/conv_0/BiasAddBiasAdd'ppo_agent/ppo2_model/pi_2/conv_0/Conv2D(ppo_agent/ppo2_model/pi/conv_0/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:�	
�
*ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu	LeakyRelu(ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd*
T0*
alpha%��L>*'
_output_shapes
:�	

.ppo_agent/ppo2_model/pi_2/conv_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
'ppo_agent/ppo2_model/pi_2/conv_1/Conv2DConv2D*ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu*ppo_agent/ppo2_model/pi/conv_1/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
(ppo_agent/ppo2_model/pi_2/conv_1/BiasAddBiasAdd'ppo_agent/ppo2_model/pi_2/conv_1/Conv2D(ppo_agent/ppo2_model/pi/conv_1/bias/read*'
_output_shapes
:�*
T0*
data_formatNHWC
�
*ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu	LeakyRelu(ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd*
alpha%��L>*'
_output_shapes
:�*
T0
�
/ppo_agent/ppo2_model/pi_2/flatten/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"�  ����
�
)ppo_agent/ppo2_model/pi_2/flatten/ReshapeReshape*ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu/ppo_agent/ppo2_model/pi_2/flatten/Reshape/shape*
T0*
Tshape0* 
_output_shapes
:
��
�
&ppo_agent/ppo2_model/pi_2/dense/MatMulMatMul)ppo_agent/ppo2_model/pi_2/flatten/Reshape)ppo_agent/ppo2_model/pi/dense/kernel/read*
transpose_b( *
T0*
_output_shapes
:	�@*
transpose_a( 
�
'ppo_agent/ppo2_model/pi_2/dense/BiasAddBiasAdd&ppo_agent/ppo2_model/pi_2/dense/MatMul'ppo_agent/ppo2_model/pi/dense/bias/read*
T0*
data_formatNHWC*
_output_shapes
:	�@
�
)ppo_agent/ppo2_model/pi_2/dense/LeakyRelu	LeakyRelu'ppo_agent/ppo2_model/pi_2/dense/BiasAdd*
alpha%��L>*
_output_shapes
:	�@*
T0
�
(ppo_agent/ppo2_model/pi_2/dense_1/MatMulMatMul)ppo_agent/ppo2_model/pi_2/dense/LeakyRelu+ppo_agent/ppo2_model/pi/dense_1/kernel/read*
T0*
_output_shapes
:	�@*
transpose_a( *
transpose_b( 
�
)ppo_agent/ppo2_model/pi_2/dense_1/BiasAddBiasAdd(ppo_agent/ppo2_model/pi_2/dense_1/MatMul)ppo_agent/ppo2_model/pi/dense_1/bias/read*
data_formatNHWC*
_output_shapes
:	�@*
T0
�
+ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu	LeakyRelu)ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd*
T0*
alpha%��L>*
_output_shapes
:	�@
�
(ppo_agent/ppo2_model/pi_2/dense_2/MatMulMatMul+ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu+ppo_agent/ppo2_model/pi/dense_2/kernel/read*
transpose_b( *
T0*
_output_shapes
:	�@*
transpose_a( 
�
)ppo_agent/ppo2_model/pi_2/dense_2/BiasAddBiasAdd(ppo_agent/ppo2_model/pi_2/dense_2/MatMul)ppo_agent/ppo2_model/pi/dense_2/bias/read*
T0*
data_formatNHWC*
_output_shapes
:	�@
�
+ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu	LeakyRelu)ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd*
_output_shapes
:	�@*
T0*
alpha%��L>
_
ppo_agent/ppo2_model/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
}
,ppo_agent/ppo2_model/flatten_2/Reshape/shapeConst*
_output_shapes
:*
valueB"�  ����*
dtype0
�
&ppo_agent/ppo2_model/flatten_2/ReshapeReshape+ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu,ppo_agent/ppo2_model/flatten_2/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	�@
}
,ppo_agent/ppo2_model/flatten_3/Reshape/shapeConst*
valueB"�  ����*
dtype0*
_output_shapes
:
�
&ppo_agent/ppo2_model/flatten_3/ReshapeReshape+ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu,ppo_agent/ppo2_model/flatten_3/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	�@
�
 ppo_agent/ppo2_model/pi_3/MatMulMatMul&ppo_agent/ppo2_model/flatten_3/Reshapeppo_agent/ppo2_model/pi/w/read*
_output_shapes
:	�*
transpose_a( *
transpose_b( *
T0
�
ppo_agent/ppo2_model/pi_3/addAdd ppo_agent/ppo2_model/pi_3/MatMulppo_agent/ppo2_model/pi/b/read*
T0*
_output_shapes
:	�
r
ppo_agent/ppo2_model/Softmax_1Softmaxppo_agent/ppo2_model/pi_3/add*
_output_shapes
:	�*
T0
y
#ppo_agent/ppo2_model/action_probs_1Identityppo_agent/ppo2_model/Softmax_1*
T0*
_output_shapes
:	�
m
ppo_agent/ppo2_model/Shape_1Const*
valueB"�     *
dtype0*
_output_shapes
:
n
)ppo_agent/ppo2_model/random_uniform_1/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n
)ppo_agent/ppo2_model/random_uniform_1/maxConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
3ppo_agent/ppo2_model/random_uniform_1/RandomUniformRandomUniformppo_agent/ppo2_model/Shape_1*
dtype0*
_output_shapes
:	�*
seed2�*
seed�*
T0
�
)ppo_agent/ppo2_model/random_uniform_1/subSub)ppo_agent/ppo2_model/random_uniform_1/max)ppo_agent/ppo2_model/random_uniform_1/min*
T0*
_output_shapes
: 
�
)ppo_agent/ppo2_model/random_uniform_1/mulMul3ppo_agent/ppo2_model/random_uniform_1/RandomUniform)ppo_agent/ppo2_model/random_uniform_1/sub*
_output_shapes
:	�*
T0
�
%ppo_agent/ppo2_model/random_uniform_1Add)ppo_agent/ppo2_model/random_uniform_1/mul)ppo_agent/ppo2_model/random_uniform_1/min*
_output_shapes
:	�*
T0
r
ppo_agent/ppo2_model/Log_2Log%ppo_agent/ppo2_model/random_uniform_1*
T0*
_output_shapes
:	�
g
ppo_agent/ppo2_model/Neg_1Negppo_agent/ppo2_model/Log_2*
T0*
_output_shapes
:	�
g
ppo_agent/ppo2_model/Log_3Logppo_agent/ppo2_model/Neg_1*
_output_shapes
:	�*
T0
�
ppo_agent/ppo2_model/sub_1Subppo_agent/ppo2_model/pi_3/addppo_agent/ppo2_model/Log_3*
T0*
_output_shapes
:	�
r
'ppo_agent/ppo2_model/ArgMax_1/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
ppo_agent/ppo2_model/ArgMax_1ArgMaxppo_agent/ppo2_model/sub_1'ppo_agent/ppo2_model/ArgMax_1/dimension*
_output_shapes	
:�*

Tidx0*
T0*
output_type0	
n
ppo_agent/ppo2_model/action_1Identityppo_agent/ppo2_model/ArgMax_1*
T0	*
_output_shapes	
:�
l
'ppo_agent/ppo2_model/one_hot_1/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
m
(ppo_agent/ppo2_model/one_hot_1/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
f
$ppo_agent/ppo2_model/one_hot_1/depthConst*
value	B :*
dtype0*
_output_shapes
: 
�
ppo_agent/ppo2_model/one_hot_1OneHotppo_agent/ppo2_model/action_1$ppo_agent/ppo2_model/one_hot_1/depth'ppo_agent/ppo2_model/one_hot_1/on_value(ppo_agent/ppo2_model/one_hot_1/off_value*
axis���������*
TI0	*
_output_shapes
:	�*
T0

=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/RankConst*
_output_shapes
: *
value	B :*
dtype0
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/ShapeConst*
valueB"�     *
dtype0*
_output_shapes
:
�
?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Shape_1Const*
valueB"�     *
dtype0*
_output_shapes
:
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/SubSub?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Rank_1>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub/y*
T0*
_output_shapes
: 
�
Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice/beginPack<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub*
T0*

axis *
N*
_output_shapes
:
�
Cppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/SliceSlice@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Shape_1Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice/beginCppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice/size*
Index0*
T0*
_output_shapes
:
�
Hppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
�
Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concatConcatV2Hppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat/values_0>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/SliceDppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/ReshapeReshapeppo_agent/ppo2_model/pi_3/add?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat*
Tshape0*
_output_shapes
:	�*
T0
�
?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Shape_2Const*
valueB"�     *
dtype0*
_output_shapes
:
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_1Sub?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Rank_2@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_1/y*
_output_shapes
: *
T0
�
Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_1/beginPack>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_1*
T0*

axis *
N*
_output_shapes
:
�
Eppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_1Slice@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Shape_2Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_1/beginEppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
Jppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
�
Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Appo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat_1ConcatV2Jppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat_1/values_0@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_1Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
Bppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Reshape_1Reshapeppo_agent/ppo2_model/one_hot_1Appo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat_1*
_output_shapes
:	�*
T0*
Tshape0
�
8ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1SoftmaxCrossEntropyWithLogits@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/ReshapeBppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Reshape_1*
T0*&
_output_shapes
:�:	�
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_2Sub=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Rank@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_2/y*
T0*
_output_shapes
: 
�
Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
Eppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_2/sizePack>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_2Slice>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/ShapeFppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_2/beginEppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_2/size*
_output_shapes
:*
Index0*
T0
�
Bppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Reshape_2Reshape8ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_2*
T0*
Tshape0*
_output_shapes	
:�
�
 ppo_agent/ppo2_model/vf_1/MatMulMatMul&ppo_agent/ppo2_model/flatten_2/Reshapeppo_agent/ppo2_model/vf/w/read*
T0*
_output_shapes
:	�*
transpose_a( *
transpose_b( 
�
ppo_agent/ppo2_model/vf_1/addAdd ppo_agent/ppo2_model/vf_1/MatMulppo_agent/ppo2_model/vf/b/read*
T0*
_output_shapes
:	�
{
*ppo_agent/ppo2_model/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
}
,ppo_agent/ppo2_model/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
}
,ppo_agent/ppo2_model/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
$ppo_agent/ppo2_model/strided_slice_1StridedSliceppo_agent/ppo2_model/vf_1/add*ppo_agent/ppo2_model/strided_slice_1/stack,ppo_agent/ppo2_model/strided_slice_1/stack_1,ppo_agent/ppo2_model/strided_slice_1/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
Index0*
T0*
shrink_axis_mask
t
ppo_agent/ppo2_model/value_1Identity$ppo_agent/ppo2_model/strided_slice_1*
T0*
_output_shapes	
:�
f
PlaceholderPlaceholder*
shape:���������*
dtype0*#
_output_shapes
:���������
h
Placeholder_1Placeholder*
shape:���������*
dtype0*#
_output_shapes
:���������
h
Placeholder_2Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
h
Placeholder_4Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
N
Placeholder_5Placeholder*
dtype0*
_output_shapes
: *
shape: 
N
Placeholder_6Placeholder*
dtype0*
_output_shapes
: *
shape: 
U
one_hot/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
V
one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
O
one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
�
one_hotOneHotPlaceholderone_hot/depthone_hot/on_valueone_hot/off_value*
T0*
axis���������*
TI0*'
_output_shapes
:���������
h
&softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
x
'softmax_cross_entropy_with_logits/ShapeConst*
valueB"�     *
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
z
)softmax_cross_entropy_with_logits/Shape_1Const*
valueB"�     *
dtype0*
_output_shapes
:
i
'softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
%softmax_cross_entropy_with_logits/SubSub(softmax_cross_entropy_with_logits/Rank_1'softmax_cross_entropy_with_logits/Sub/y*
_output_shapes
: *
T0
�
-softmax_cross_entropy_with_logits/Slice/beginPack%softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N*
_output_shapes
:
v
,softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
'softmax_cross_entropy_with_logits/SliceSlice)softmax_cross_entropy_with_logits/Shape_1-softmax_cross_entropy_with_logits/Slice/begin,softmax_cross_entropy_with_logits/Slice/size*
_output_shapes
:*
Index0*
T0
�
1softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
o
-softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
(softmax_cross_entropy_with_logits/concatConcatV21softmax_cross_entropy_with_logits/concat/values_0'softmax_cross_entropy_with_logits/Slice-softmax_cross_entropy_with_logits/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
)softmax_cross_entropy_with_logits/ReshapeReshapeppo_agent/ppo2_model/pi_3/add(softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*
_output_shapes
:	�
j
(softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
p
)softmax_cross_entropy_with_logits/Shape_2Shapeone_hot*
T0*
out_type0*
_output_shapes
:
k
)softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'softmax_cross_entropy_with_logits/Sub_1Sub(softmax_cross_entropy_with_logits/Rank_2)softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
�
/softmax_cross_entropy_with_logits/Slice_1/beginPack'softmax_cross_entropy_with_logits/Sub_1*
T0*

axis *
N*
_output_shapes
:
x
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
)softmax_cross_entropy_with_logits/Slice_1Slice)softmax_cross_entropy_with_logits/Shape_2/softmax_cross_entropy_with_logits/Slice_1/begin.softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
3softmax_cross_entropy_with_logits/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
q
/softmax_cross_entropy_with_logits/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
*softmax_cross_entropy_with_logits/concat_1ConcatV23softmax_cross_entropy_with_logits/concat_1/values_0)softmax_cross_entropy_with_logits/Slice_1/softmax_cross_entropy_with_logits/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
+softmax_cross_entropy_with_logits/Reshape_1Reshapeone_hot*softmax_cross_entropy_with_logits/concat_1*0
_output_shapes
:������������������*
T0*
Tshape0
�
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits)softmax_cross_entropy_with_logits/Reshape+softmax_cross_entropy_with_logits/Reshape_1*
T0*&
_output_shapes
:�:	�
k
)softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'softmax_cross_entropy_with_logits/Sub_2Sub&softmax_cross_entropy_with_logits/Rank)softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
y
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
.softmax_cross_entropy_with_logits/Slice_2/sizePack'softmax_cross_entropy_with_logits/Sub_2*

axis *
N*
_output_shapes
:*
T0
�
)softmax_cross_entropy_with_logits/Slice_2Slice'softmax_cross_entropy_with_logits/Shape/softmax_cross_entropy_with_logits/Slice_2/begin.softmax_cross_entropy_with_logits/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
+softmax_cross_entropy_with_logits/Reshape_2Reshape!softmax_cross_entropy_with_logits)softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0*
_output_shapes	
:�
`
Max/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
MaxMaxppo_agent/ppo2_model/pi_3/addMax/reduction_indices*
T0*
_output_shapes
:	�*
	keep_dims(*

Tidx0
X
subSubppo_agent/ppo2_model/pi_3/addMax*
T0*
_output_shapes
:	�
9
ExpExpsub*
_output_shapes
:	�*
T0
`
Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
���������
m
SumSumExpSum/reduction_indices*
_output_shapes
:	�*
	keep_dims(*

Tidx0*
T0
F
truedivRealDivExpSum*
T0*
_output_shapes
:	�
9
LogLogSum*
T0*
_output_shapes
:	�
@
sub_1SubLogsub*
T0*
_output_shapes
:	�
D
mulMultruedivsub_1*
_output_shapes
:	�*
T0
b
Sum_1/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
���������
m
Sum_1SummulSum_1/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
X
MeanMeanSum_1Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
_
sub_2Subppo_agent/ppo2_model/value_1Placeholder_4*
T0*
_output_shapes	
:�
:
NegNegPlaceholder_6*
T0*
_output_shapes
: 
\
clip_by_value/MinimumMinimumsub_2Placeholder_6*
T0*
_output_shapes	
:�
Z
clip_by_valueMaximumclip_by_value/MinimumNeg*
_output_shapes	
:�*
T0
N
addAddPlaceholder_4clip_by_value*
T0*
_output_shapes	
:�
_
sub_3Subppo_agent/ppo2_model/value_1Placeholder_2*
_output_shapes	
:�*
T0
=
SquareSquaresub_3*
T0*
_output_shapes	
:�
F
sub_4SubaddPlaceholder_2*
_output_shapes	
:�*
T0
?
Square_1Squaresub_4*
T0*
_output_shapes	
:�
J
MaximumMaximumSquareSquare_1*
T0*
_output_shapes	
:�
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
^
Mean_1MeanMaximumConst_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
mul_1/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
>
mul_1Mulmul_1/xMean_1*
_output_shapes
: *
T0
n
sub_5SubPlaceholder_3+softmax_cross_entropy_with_logits/Reshape_2*
T0*
_output_shapes	
:�
9
Exp_1Expsub_5*
T0*
_output_shapes	
:�
I
Neg_1NegPlaceholder_1*
T0*#
_output_shapes
:���������
@
mul_2MulNeg_1Exp_1*
T0*
_output_shapes	
:�
I
Neg_2NegPlaceholder_1*
T0*#
_output_shapes
:���������
L
sub_6/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
E
sub_6Subsub_6/xPlaceholder_6*
T0*
_output_shapes
: 
L
add_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
E
add_1Addadd_1/xPlaceholder_6*
T0*
_output_shapes
: 
V
clip_by_value_1/MinimumMinimumExp_1add_1*
_output_shapes	
:�*
T0
`
clip_by_value_1Maximumclip_by_value_1/Minimumsub_6*
_output_shapes	
:�*
T0
J
mul_3MulNeg_2clip_by_value_1*
T0*
_output_shapes	
:�
H
	Maximum_1Maximummul_2mul_3*
_output_shapes	
:�*
T0
Q
Const_2Const*
_output_shapes
:*
valueB: *
dtype0
`
Mean_2Mean	Maximum_1Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
n
sub_7Sub+softmax_cross_entropy_with_logits/Reshape_2Placeholder_3*
T0*
_output_shapes	
:�
?
Square_2Squaresub_7*
_output_shapes	
:�*
T0
Q
Const_3Const*
valueB: *
dtype0*
_output_shapes
:
_
Mean_3MeanSquare_2Const_3*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
L
mul_4/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
>
mul_4Mulmul_4/xMean_3*
_output_shapes
: *
T0
L
sub_8/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
B
sub_8SubExp_1sub_8/y*
_output_shapes	
:�*
T0
7
AbsAbssub_8*
T0*
_output_shapes	
:�
L
GreaterGreaterAbsPlaceholder_6*
T0*
_output_shapes	
:�
]
ToFloatCastGreater*

SrcT0
*
Truncate( *
_output_shapes	
:�*

DstT0
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:
^
Mean_4MeanToFloatConst_4*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
mul_5/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
<
mul_5MulMeanmul_5/y*
_output_shapes
: *
T0
<
sub_9SubMean_2mul_5*
T0*
_output_shapes
: 
L
mul_6/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
=
mul_6Mulmul_1mul_6/y*
T0*
_output_shapes
: 
;
add_2Addsub_9mul_6*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
>
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/Fill
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/Fill&^gradients/add_2_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/Fill&^gradients/add_2_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
o
gradients/sub_9_grad/NegNeg-gradients/add_2_grad/tuple/control_dependency*
_output_shapes
: *
T0
x
%gradients/sub_9_grad/tuple/group_depsNoOp.^gradients/add_2_grad/tuple/control_dependency^gradients/sub_9_grad/Neg
�
-gradients/sub_9_grad/tuple/control_dependencyIdentity-gradients/add_2_grad/tuple/control_dependency&^gradients/sub_9_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
_output_shapes
: *
T0
�
/gradients/sub_9_grad/tuple/control_dependency_1Identitygradients/sub_9_grad/Neg&^gradients/sub_9_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/sub_9_grad/Neg
z
gradients/mul_6_grad/MulMul/gradients/add_2_grad/tuple/control_dependency_1mul_6/y*
T0*
_output_shapes
: 
z
gradients/mul_6_grad/Mul_1Mul/gradients/add_2_grad/tuple/control_dependency_1mul_1*
_output_shapes
: *
T0
e
%gradients/mul_6_grad/tuple/group_depsNoOp^gradients/mul_6_grad/Mul^gradients/mul_6_grad/Mul_1
�
-gradients/mul_6_grad/tuple/control_dependencyIdentitygradients/mul_6_grad/Mul&^gradients/mul_6_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/mul_6_grad/Mul
�
/gradients/mul_6_grad/tuple/control_dependency_1Identitygradients/mul_6_grad/Mul_1&^gradients/mul_6_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_6_grad/Mul_1*
_output_shapes
: 
m
#gradients/Mean_2_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_2_grad/ReshapeReshape-gradients/sub_9_grad/tuple/control_dependency#gradients/Mean_2_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
f
gradients/Mean_2_grad/ConstConst*
valueB:�*
dtype0*
_output_shapes
:
�
gradients/Mean_2_grad/TileTilegradients/Mean_2_grad/Reshapegradients/Mean_2_grad/Const*

Tmultiples0*
T0*
_output_shapes	
:�
b
gradients/Mean_2_grad/Const_1Const*
_output_shapes
: *
valueB
 *  zD*
dtype0
�
gradients/Mean_2_grad/truedivRealDivgradients/Mean_2_grad/Tilegradients/Mean_2_grad/Const_1*
_output_shapes	
:�*
T0
z
gradients/mul_5_grad/MulMul/gradients/sub_9_grad/tuple/control_dependency_1mul_5/y*
T0*
_output_shapes
: 
y
gradients/mul_5_grad/Mul_1Mul/gradients/sub_9_grad/tuple/control_dependency_1Mean*
T0*
_output_shapes
: 
e
%gradients/mul_5_grad/tuple/group_depsNoOp^gradients/mul_5_grad/Mul^gradients/mul_5_grad/Mul_1
�
-gradients/mul_5_grad/tuple/control_dependencyIdentitygradients/mul_5_grad/Mul&^gradients/mul_5_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_5_grad/Mul*
_output_shapes
: 
�
/gradients/mul_5_grad/tuple/control_dependency_1Identitygradients/mul_5_grad/Mul_1&^gradients/mul_5_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_5_grad/Mul_1*
_output_shapes
: 
w
gradients/mul_1_grad/MulMul-gradients/mul_6_grad/tuple/control_dependencyMean_1*
_output_shapes
: *
T0
z
gradients/mul_1_grad/Mul_1Mul-gradients/mul_6_grad/tuple/control_dependencymul_1/x*
T0*
_output_shapes
: 
e
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Mul^gradients/mul_1_grad/Mul_1
�
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Mul&^gradients/mul_1_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
�
/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Mul_1&^gradients/mul_1_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_1_grad/Mul_1*
_output_shapes
: *
T0
i
gradients/Maximum_1_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
k
 gradients/Maximum_1_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
k
 gradients/Maximum_1_grad/Shape_2Const*
valueB:�*
dtype0*
_output_shapes
:
i
$gradients/Maximum_1_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
gradients/Maximum_1_grad/zerosFill gradients/Maximum_1_grad/Shape_2$gradients/Maximum_1_grad/zeros/Const*
T0*

index_type0*
_output_shapes	
:�
i
%gradients/Maximum_1_grad/GreaterEqualGreaterEqualmul_2mul_3*
T0*
_output_shapes	
:�
�
.gradients/Maximum_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_1_grad/Shape gradients/Maximum_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Maximum_1_grad/SelectSelect%gradients/Maximum_1_grad/GreaterEqualgradients/Mean_2_grad/truedivgradients/Maximum_1_grad/zeros*
T0*
_output_shapes	
:�
�
!gradients/Maximum_1_grad/Select_1Select%gradients/Maximum_1_grad/GreaterEqualgradients/Maximum_1_grad/zerosgradients/Mean_2_grad/truediv*
T0*
_output_shapes	
:�
�
gradients/Maximum_1_grad/SumSumgradients/Maximum_1_grad/Select.gradients/Maximum_1_grad/BroadcastGradientArgs*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
 gradients/Maximum_1_grad/ReshapeReshapegradients/Maximum_1_grad/Sumgradients/Maximum_1_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
gradients/Maximum_1_grad/Sum_1Sum!gradients/Maximum_1_grad/Select_10gradients/Maximum_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
"gradients/Maximum_1_grad/Reshape_1Reshapegradients/Maximum_1_grad/Sum_1 gradients/Maximum_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
y
)gradients/Maximum_1_grad/tuple/group_depsNoOp!^gradients/Maximum_1_grad/Reshape#^gradients/Maximum_1_grad/Reshape_1
�
1gradients/Maximum_1_grad/tuple/control_dependencyIdentity gradients/Maximum_1_grad/Reshape*^gradients/Maximum_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Maximum_1_grad/Reshape*
_output_shapes	
:�
�
3gradients/Maximum_1_grad/tuple/control_dependency_1Identity"gradients/Maximum_1_grad/Reshape_1*^gradients/Maximum_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Maximum_1_grad/Reshape_1*
_output_shapes	
:�
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshape-gradients/mul_5_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
d
gradients/Mean_grad/ConstConst*
valueB:�*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Const*

Tmultiples0*
T0*
_output_shapes	
:�
`
gradients/Mean_grad/Const_1Const*
_output_shapes
: *
valueB
 *  zD*
dtype0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Const_1*
T0*
_output_shapes	
:�
m
#gradients/Mean_1_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_1_grad/ReshapeReshape/gradients/mul_1_grad/tuple/control_dependency_1#gradients/Mean_1_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
f
gradients/Mean_1_grad/ConstConst*
valueB:�*
dtype0*
_output_shapes
:
�
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Const*
_output_shapes	
:�*

Tmultiples0*
T0
b
gradients/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  zD
�
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Const_1*
_output_shapes	
:�*
T0
_
gradients/mul_2_grad/ShapeShapeNeg_1*
out_type0*
_output_shapes
:*
T0
g
gradients/mul_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������

gradients/mul_2_grad/MulMul1gradients/Maximum_1_grad/tuple/control_dependencyExp_1*
T0*
_output_shapes	
:�
�
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/mul_2_grad/Mul_1MulNeg_11gradients/Maximum_1_grad/tuple/control_dependency*
_output_shapes	
:�*
T0
�
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
Tshape0*
_output_shapes	
:�*
T0
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
�
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*#
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape
�
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1
_
gradients/mul_3_grad/ShapeShapeNeg_2*
_output_shapes
:*
T0*
out_type0
g
gradients/mul_3_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
*gradients/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_3_grad/Shapegradients/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/mul_3_grad/MulMul3gradients/Maximum_1_grad/tuple/control_dependency_1clip_by_value_1*
_output_shapes	
:�*
T0
�
gradients/mul_3_grad/SumSumgradients/mul_3_grad/Mul*gradients/mul_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sumgradients/mul_3_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/mul_3_grad/Mul_1MulNeg_23gradients/Maximum_1_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0
�
gradients/mul_3_grad/Sum_1Sumgradients/mul_3_grad/Mul_1,gradients/mul_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/mul_3_grad/Reshape_1Reshapegradients/mul_3_grad/Sum_1gradients/mul_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
m
%gradients/mul_3_grad/tuple/group_depsNoOp^gradients/mul_3_grad/Reshape^gradients/mul_3_grad/Reshape_1
�
-gradients/mul_3_grad/tuple/control_dependencyIdentitygradients/mul_3_grad/Reshape&^gradients/mul_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_3_grad/Reshape*#
_output_shapes
:���������
�
/gradients/mul_3_grad/tuple/control_dependency_1Identitygradients/mul_3_grad/Reshape_1&^gradients/mul_3_grad/tuple/group_deps*
_output_shapes	
:�*
T0*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1
k
gradients/Sum_1_grad/ShapeConst*
valueB"�     *
dtype0*
_output_shapes
:
�
gradients/Sum_1_grad/SizeConst*
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
�
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
�
gradients/Sum_1_grad/Shape_1Const*
valueB *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
 gradients/Sum_1_grad/range/startConst*
value	B : *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
 gradients/Sum_1_grad/range/deltaConst*
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:*

Tidx0
�
gradients/Sum_1_grad/Fill/valueConst*
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*

index_type0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
�
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_1_grad/Maximum/yConst*
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
�
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
�
gradients/Sum_1_grad/ReshapeReshapegradients/Mean_grad/truediv"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:	�
g
gradients/Maximum_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:�
i
gradients/Maximum_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
i
gradients/Maximum_grad/Shape_2Const*
valueB:�*
dtype0*
_output_shapes
:
g
"gradients/Maximum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/Maximum_grad/zerosFillgradients/Maximum_grad/Shape_2"gradients/Maximum_grad/zeros/Const*

index_type0*
_output_shapes	
:�*
T0
k
#gradients/Maximum_grad/GreaterEqualGreaterEqualSquareSquare_1*
T0*
_output_shapes	
:�
�
,gradients/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_grad/Shapegradients/Maximum_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Maximum_grad/SelectSelect#gradients/Maximum_grad/GreaterEqualgradients/Mean_1_grad/truedivgradients/Maximum_grad/zeros*
T0*
_output_shapes	
:�
�
gradients/Maximum_grad/Select_1Select#gradients/Maximum_grad/GreaterEqualgradients/Maximum_grad/zerosgradients/Mean_1_grad/truediv*
_output_shapes	
:�*
T0
�
gradients/Maximum_grad/SumSumgradients/Maximum_grad/Select,gradients/Maximum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
gradients/Maximum_grad/Sum_1Sumgradients/Maximum_grad/Select_1.gradients/Maximum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
 gradients/Maximum_grad/Reshape_1Reshapegradients/Maximum_grad/Sum_1gradients/Maximum_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
s
'gradients/Maximum_grad/tuple/group_depsNoOp^gradients/Maximum_grad/Reshape!^gradients/Maximum_grad/Reshape_1
�
/gradients/Maximum_grad/tuple/control_dependencyIdentitygradients/Maximum_grad/Reshape(^gradients/Maximum_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Maximum_grad/Reshape*
_output_shapes	
:�
�
1gradients/Maximum_grad/tuple/control_dependency_1Identity gradients/Maximum_grad/Reshape_1(^gradients/Maximum_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Maximum_grad/Reshape_1*
_output_shapes	
:�
o
$gradients/clip_by_value_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:�
i
&gradients/clip_by_value_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
q
&gradients/clip_by_value_1_grad/Shape_2Const*
valueB:�*
dtype0*
_output_shapes
:
o
*gradients/clip_by_value_1_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$gradients/clip_by_value_1_grad/zerosFill&gradients/clip_by_value_1_grad/Shape_2*gradients/clip_by_value_1_grad/zeros/Const*
T0*

index_type0*
_output_shapes	
:�
�
+gradients/clip_by_value_1_grad/GreaterEqualGreaterEqualclip_by_value_1/Minimumsub_6*
_output_shapes	
:�*
T0
�
4gradients/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/clip_by_value_1_grad/Shape&gradients/clip_by_value_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
%gradients/clip_by_value_1_grad/SelectSelect+gradients/clip_by_value_1_grad/GreaterEqual/gradients/mul_3_grad/tuple/control_dependency_1$gradients/clip_by_value_1_grad/zeros*
_output_shapes	
:�*
T0
�
'gradients/clip_by_value_1_grad/Select_1Select+gradients/clip_by_value_1_grad/GreaterEqual$gradients/clip_by_value_1_grad/zeros/gradients/mul_3_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
"gradients/clip_by_value_1_grad/SumSum%gradients/clip_by_value_1_grad/Select4gradients/clip_by_value_1_grad/BroadcastGradientArgs*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
&gradients/clip_by_value_1_grad/ReshapeReshape"gradients/clip_by_value_1_grad/Sum$gradients/clip_by_value_1_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
$gradients/clip_by_value_1_grad/Sum_1Sum'gradients/clip_by_value_1_grad/Select_16gradients/clip_by_value_1_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
(gradients/clip_by_value_1_grad/Reshape_1Reshape$gradients/clip_by_value_1_grad/Sum_1&gradients/clip_by_value_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
/gradients/clip_by_value_1_grad/tuple/group_depsNoOp'^gradients/clip_by_value_1_grad/Reshape)^gradients/clip_by_value_1_grad/Reshape_1
�
7gradients/clip_by_value_1_grad/tuple/control_dependencyIdentity&gradients/clip_by_value_1_grad/Reshape0^gradients/clip_by_value_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/clip_by_value_1_grad/Reshape*
_output_shapes	
:�
�
9gradients/clip_by_value_1_grad/tuple/control_dependency_1Identity(gradients/clip_by_value_1_grad/Reshape_10^gradients/clip_by_value_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/clip_by_value_1_grad/Reshape_1*
_output_shapes
: 
i
gradients/mul_grad/MulMulgradients/Sum_1_grad/Tilesub_1*
_output_shapes
:	�*
T0
m
gradients/mul_grad/Mul_1Mulgradients/Sum_1_grad/Tiletruediv*
T0*
_output_shapes
:	�
_
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Mul^gradients/mul_grad/Mul_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Mul$^gradients/mul_grad/tuple/group_deps*
T0*)
_class
loc:@gradients/mul_grad/Mul*
_output_shapes
:	�
�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Mul_1$^gradients/mul_grad/tuple/group_deps*+
_class!
loc:@gradients/mul_grad/Mul_1*
_output_shapes
:	�*
T0
�
gradients/Square_grad/ConstConst0^gradients/Maximum_grad/tuple/control_dependency*
dtype0*
_output_shapes
: *
valueB
 *   @
j
gradients/Square_grad/MulMulsub_3gradients/Square_grad/Const*
_output_shapes	
:�*
T0
�
gradients/Square_grad/Mul_1Mul/gradients/Maximum_grad/tuple/control_dependencygradients/Square_grad/Mul*
T0*
_output_shapes	
:�
�
gradients/Square_1_grad/ConstConst2^gradients/Maximum_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
n
gradients/Square_1_grad/MulMulsub_4gradients/Square_1_grad/Const*
T0*
_output_shapes	
:�
�
gradients/Square_1_grad/Mul_1Mul1gradients/Maximum_grad/tuple/control_dependency_1gradients/Square_1_grad/Mul*
T0*
_output_shapes	
:�
w
,gradients/clip_by_value_1/Minimum_grad/ShapeConst*
_output_shapes
:*
valueB:�*
dtype0
q
.gradients/clip_by_value_1/Minimum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
y
.gradients/clip_by_value_1/Minimum_grad/Shape_2Const*
_output_shapes
:*
valueB:�*
dtype0
w
2gradients/clip_by_value_1/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
,gradients/clip_by_value_1/Minimum_grad/zerosFill.gradients/clip_by_value_1/Minimum_grad/Shape_22gradients/clip_by_value_1/Minimum_grad/zeros/Const*
T0*

index_type0*
_output_shapes	
:�
q
0gradients/clip_by_value_1/Minimum_grad/LessEqual	LessEqualExp_1add_1*
_output_shapes	
:�*
T0
�
<gradients/clip_by_value_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/clip_by_value_1/Minimum_grad/Shape.gradients/clip_by_value_1/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
-gradients/clip_by_value_1/Minimum_grad/SelectSelect0gradients/clip_by_value_1/Minimum_grad/LessEqual7gradients/clip_by_value_1_grad/tuple/control_dependency,gradients/clip_by_value_1/Minimum_grad/zeros*
T0*
_output_shapes	
:�
�
/gradients/clip_by_value_1/Minimum_grad/Select_1Select0gradients/clip_by_value_1/Minimum_grad/LessEqual,gradients/clip_by_value_1/Minimum_grad/zeros7gradients/clip_by_value_1_grad/tuple/control_dependency*
T0*
_output_shapes	
:�
�
*gradients/clip_by_value_1/Minimum_grad/SumSum-gradients/clip_by_value_1/Minimum_grad/Select<gradients/clip_by_value_1/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
.gradients/clip_by_value_1/Minimum_grad/ReshapeReshape*gradients/clip_by_value_1/Minimum_grad/Sum,gradients/clip_by_value_1/Minimum_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
,gradients/clip_by_value_1/Minimum_grad/Sum_1Sum/gradients/clip_by_value_1/Minimum_grad/Select_1>gradients/clip_by_value_1/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
0gradients/clip_by_value_1/Minimum_grad/Reshape_1Reshape,gradients/clip_by_value_1/Minimum_grad/Sum_1.gradients/clip_by_value_1/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
7gradients/clip_by_value_1/Minimum_grad/tuple/group_depsNoOp/^gradients/clip_by_value_1/Minimum_grad/Reshape1^gradients/clip_by_value_1/Minimum_grad/Reshape_1
�
?gradients/clip_by_value_1/Minimum_grad/tuple/control_dependencyIdentity.gradients/clip_by_value_1/Minimum_grad/Reshape8^gradients/clip_by_value_1/Minimum_grad/tuple/group_deps*
_output_shapes	
:�*
T0*A
_class7
53loc:@gradients/clip_by_value_1/Minimum_grad/Reshape
�
Agradients/clip_by_value_1/Minimum_grad/tuple/control_dependency_1Identity0gradients/clip_by_value_1/Minimum_grad/Reshape_18^gradients/clip_by_value_1/Minimum_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/clip_by_value_1/Minimum_grad/Reshape_1*
_output_shapes
: 
m
gradients/truediv_grad/ShapeConst*
valueB"�     *
dtype0*
_output_shapes
:
o
gradients/truediv_grad/Shape_1Const*
valueB"�     *
dtype0*
_output_shapes
:
�
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/truediv_grad/RealDivRealDiv+gradients/mul_grad/tuple/control_dependencySum*
T0*
_output_shapes
:	�
�
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	�
�
gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
_output_shapes
:	�*
T0*
Tshape0
P
gradients/truediv_grad/NegNegExp*
_output_shapes
:	�*
T0
v
 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/NegSum*
_output_shapes
:	�*
T0
|
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1Sum*
_output_shapes
:	�*
T0
�
gradients/truediv_grad/mulMul+gradients/mul_grad/tuple/control_dependency gradients/truediv_grad/RealDiv_2*
_output_shapes
:	�*
T0
�
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:	�
s
'gradients/truediv_grad/tuple/group_depsNoOp^gradients/truediv_grad/Reshape!^gradients/truediv_grad/Reshape_1
�
/gradients/truediv_grad/tuple/control_dependencyIdentitygradients/truediv_grad/Reshape(^gradients/truediv_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/truediv_grad/Reshape*
_output_shapes
:	�
�
1gradients/truediv_grad/tuple/control_dependency_1Identity gradients/truediv_grad/Reshape_1(^gradients/truediv_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_grad/Reshape_1*
_output_shapes
:	�
k
gradients/sub_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�     
m
gradients/sub_1_grad/Shape_1Const*
valueB"�     *
dtype0*
_output_shapes
:
�
*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_1_grad/SumSum-gradients/mul_grad/tuple/control_dependency_1*gradients/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/sub_1_grad/Sum_1Sum-gradients/mul_grad/tuple/control_dependency_1,gradients/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:	�*
	keep_dims( *

Tidx0
e
gradients/sub_1_grad/NegNeggradients/sub_1_grad/Sum_1*
_output_shapes
:	�*
T0
�
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*
_output_shapes
:	�*
T0*
Tshape0
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
�
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape*
_output_shapes
:	�
�
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*
_output_shapes
:	�*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1
e
gradients/sub_3_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
i
gradients/sub_3_grad/Shape_1ShapePlaceholder_2*
out_type0*
_output_shapes
:*
T0
�
*gradients/sub_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_3_grad/Shapegradients/sub_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_3_grad/SumSumgradients/Square_grad/Mul_1*gradients/sub_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/sub_3_grad/ReshapeReshapegradients/sub_3_grad/Sumgradients/sub_3_grad/Shape*
_output_shapes	
:�*
T0*
Tshape0
�
gradients/sub_3_grad/Sum_1Sumgradients/Square_grad/Mul_1,gradients/sub_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
^
gradients/sub_3_grad/NegNeggradients/sub_3_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_3_grad/Reshape_1Reshapegradients/sub_3_grad/Neggradients/sub_3_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
m
%gradients/sub_3_grad/tuple/group_depsNoOp^gradients/sub_3_grad/Reshape^gradients/sub_3_grad/Reshape_1
�
-gradients/sub_3_grad/tuple/control_dependencyIdentitygradients/sub_3_grad/Reshape&^gradients/sub_3_grad/tuple/group_deps*
_output_shapes	
:�*
T0*/
_class%
#!loc:@gradients/sub_3_grad/Reshape
�
/gradients/sub_3_grad/tuple/control_dependency_1Identitygradients/sub_3_grad/Reshape_1&^gradients/sub_3_grad/tuple/group_deps*#
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/sub_3_grad/Reshape_1
e
gradients/sub_4_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
i
gradients/sub_4_grad/Shape_1ShapePlaceholder_2*
T0*
out_type0*
_output_shapes
:
�
*gradients/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_4_grad/Shapegradients/sub_4_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/sub_4_grad/SumSumgradients/Square_1_grad/Mul_1*gradients/sub_4_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/sub_4_grad/ReshapeReshapegradients/sub_4_grad/Sumgradients/sub_4_grad/Shape*
_output_shapes	
:�*
T0*
Tshape0
�
gradients/sub_4_grad/Sum_1Sumgradients/Square_1_grad/Mul_1,gradients/sub_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
^
gradients/sub_4_grad/NegNeggradients/sub_4_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_4_grad/Reshape_1Reshapegradients/sub_4_grad/Neggradients/sub_4_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0
m
%gradients/sub_4_grad/tuple/group_depsNoOp^gradients/sub_4_grad/Reshape^gradients/sub_4_grad/Reshape_1
�
-gradients/sub_4_grad/tuple/control_dependencyIdentitygradients/sub_4_grad/Reshape&^gradients/sub_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_4_grad/Reshape*
_output_shapes	
:�
�
/gradients/sub_4_grad/tuple/control_dependency_1Identitygradients/sub_4_grad/Reshape_1&^gradients/sub_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_4_grad/Reshape_1*#
_output_shapes
:���������
�
gradients/AddNAddN/gradients/mul_2_grad/tuple/control_dependency_1?gradients/clip_by_value_1/Minimum_grad/tuple/control_dependency*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*
N*
_output_shapes	
:�
\
gradients/Exp_1_grad/mulMulgradients/AddNExp_1*
_output_shapes	
:�*
T0
�
gradients/Log_grad/Reciprocal
ReciprocalSum.^gradients/sub_1_grad/tuple/control_dependency*
T0*
_output_shapes
:	�
�
gradients/Log_grad/mulMul-gradients/sub_1_grad/tuple/control_dependencygradients/Log_grad/Reciprocal*
T0*
_output_shapes
:	�
e
gradients/add_grad/ShapeShapePlaceholder_4*
_output_shapes
:*
T0*
out_type0
e
gradients/add_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSum-gradients/sub_4_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sum-gradients/sub_4_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*#
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes	
:�
g
gradients/sub_5_grad/ShapeShapePlaceholder_3*
out_type0*
_output_shapes
:*
T0
g
gradients/sub_5_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
*gradients/sub_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_5_grad/Shapegradients/sub_5_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/sub_5_grad/SumSumgradients/Exp_1_grad/mul*gradients/sub_5_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/sub_5_grad/ReshapeReshapegradients/sub_5_grad/Sumgradients/sub_5_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/sub_5_grad/Sum_1Sumgradients/Exp_1_grad/mul,gradients/sub_5_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
^
gradients/sub_5_grad/NegNeggradients/sub_5_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_5_grad/Reshape_1Reshapegradients/sub_5_grad/Neggradients/sub_5_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
m
%gradients/sub_5_grad/tuple/group_depsNoOp^gradients/sub_5_grad/Reshape^gradients/sub_5_grad/Reshape_1
�
-gradients/sub_5_grad/tuple/control_dependencyIdentitygradients/sub_5_grad/Reshape&^gradients/sub_5_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_5_grad/Reshape*#
_output_shapes
:���������
�
/gradients/sub_5_grad/tuple/control_dependency_1Identitygradients/sub_5_grad/Reshape_1&^gradients/sub_5_grad/tuple/group_deps*
_output_shapes	
:�*
T0*1
_class'
%#loc:@gradients/sub_5_grad/Reshape_1
�
gradients/AddN_1AddN1gradients/truediv_grad/tuple/control_dependency_1gradients/Log_grad/mul*
_output_shapes
:	�*
T0*3
_class)
'%loc:@gradients/truediv_grad/Reshape_1*
N
i
gradients/Sum_grad/ShapeConst*
valueB"�     *
dtype0*
_output_shapes
:
�
gradients/Sum_grad/SizeConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
�
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
�
gradients/Sum_grad/Shape_1Const*
valueB *+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/range/startConst*
_output_shapes
: *
value	B : *+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0
�
gradients/Sum_grad/range/deltaConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
�
gradients/Sum_grad/Fill/valueConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*

index_type0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
T0
�
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_grad/Maximum/yConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
_output_shapes
:*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
�
gradients/Sum_grad/ReshapeReshapegradients/AddN_1 gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
T0*
_output_shapes
:	�*

Tmultiples0
m
"gradients/clip_by_value_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
g
$gradients/clip_by_value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
o
$gradients/clip_by_value_grad/Shape_2Const*
valueB:�*
dtype0*
_output_shapes
:
m
(gradients/clip_by_value_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
"gradients/clip_by_value_grad/zerosFill$gradients/clip_by_value_grad/Shape_2(gradients/clip_by_value_grad/zeros/Const*
T0*

index_type0*
_output_shapes	
:�
{
)gradients/clip_by_value_grad/GreaterEqualGreaterEqualclip_by_value/MinimumNeg*
T0*
_output_shapes	
:�
�
2gradients/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/clip_by_value_grad/Shape$gradients/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#gradients/clip_by_value_grad/SelectSelect)gradients/clip_by_value_grad/GreaterEqual-gradients/add_grad/tuple/control_dependency_1"gradients/clip_by_value_grad/zeros*
_output_shapes	
:�*
T0
�
%gradients/clip_by_value_grad/Select_1Select)gradients/clip_by_value_grad/GreaterEqual"gradients/clip_by_value_grad/zeros-gradients/add_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
 gradients/clip_by_value_grad/SumSum#gradients/clip_by_value_grad/Select2gradients/clip_by_value_grad/BroadcastGradientArgs*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
$gradients/clip_by_value_grad/ReshapeReshape gradients/clip_by_value_grad/Sum"gradients/clip_by_value_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
"gradients/clip_by_value_grad/Sum_1Sum%gradients/clip_by_value_grad/Select_14gradients/clip_by_value_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
&gradients/clip_by_value_grad/Reshape_1Reshape"gradients/clip_by_value_grad/Sum_1$gradients/clip_by_value_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
-gradients/clip_by_value_grad/tuple/group_depsNoOp%^gradients/clip_by_value_grad/Reshape'^gradients/clip_by_value_grad/Reshape_1
�
5gradients/clip_by_value_grad/tuple/control_dependencyIdentity$gradients/clip_by_value_grad/Reshape.^gradients/clip_by_value_grad/tuple/group_deps*
_output_shapes	
:�*
T0*7
_class-
+)loc:@gradients/clip_by_value_grad/Reshape
�
7gradients/clip_by_value_grad/tuple/control_dependency_1Identity&gradients/clip_by_value_grad/Reshape_1.^gradients/clip_by_value_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/clip_by_value_grad/Reshape_1*
_output_shapes
: 
�
@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
Bgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshape/gradients/sub_5_grad/tuple/control_dependency_1@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
Tshape0*
_output_shapes	
:�*
T0
�
gradients/AddN_2AddN/gradients/truediv_grad/tuple/control_dependencygradients/Sum_grad/Tile*
T0*1
_class'
%#loc:@gradients/truediv_grad/Reshape*
N*
_output_shapes
:	�
^
gradients/Exp_grad/mulMulgradients/AddN_2Exp*
T0*
_output_shapes
:	�
u
*gradients/clip_by_value/Minimum_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
o
,gradients/clip_by_value/Minimum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
w
,gradients/clip_by_value/Minimum_grad/Shape_2Const*
valueB:�*
dtype0*
_output_shapes
:
u
0gradients/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
*gradients/clip_by_value/Minimum_grad/zerosFill,gradients/clip_by_value/Minimum_grad/Shape_20gradients/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0*
_output_shapes	
:�
w
.gradients/clip_by_value/Minimum_grad/LessEqual	LessEqualsub_2Placeholder_6*
T0*
_output_shapes	
:�
�
:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/clip_by_value/Minimum_grad/Shape,gradients/clip_by_value/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
+gradients/clip_by_value/Minimum_grad/SelectSelect.gradients/clip_by_value/Minimum_grad/LessEqual5gradients/clip_by_value_grad/tuple/control_dependency*gradients/clip_by_value/Minimum_grad/zeros*
T0*
_output_shapes	
:�
�
-gradients/clip_by_value/Minimum_grad/Select_1Select.gradients/clip_by_value/Minimum_grad/LessEqual*gradients/clip_by_value/Minimum_grad/zeros5gradients/clip_by_value_grad/tuple/control_dependency*
_output_shapes	
:�*
T0
�
(gradients/clip_by_value/Minimum_grad/SumSum+gradients/clip_by_value/Minimum_grad/Select:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
,gradients/clip_by_value/Minimum_grad/ReshapeReshape(gradients/clip_by_value/Minimum_grad/Sum*gradients/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
*gradients/clip_by_value/Minimum_grad/Sum_1Sum-gradients/clip_by_value/Minimum_grad/Select_1<gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
.gradients/clip_by_value/Minimum_grad/Reshape_1Reshape*gradients/clip_by_value/Minimum_grad/Sum_1,gradients/clip_by_value/Minimum_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
5gradients/clip_by_value/Minimum_grad/tuple/group_depsNoOp-^gradients/clip_by_value/Minimum_grad/Reshape/^gradients/clip_by_value/Minimum_grad/Reshape_1
�
=gradients/clip_by_value/Minimum_grad/tuple/control_dependencyIdentity,gradients/clip_by_value/Minimum_grad/Reshape6^gradients/clip_by_value/Minimum_grad/tuple/group_deps*?
_class5
31loc:@gradients/clip_by_value/Minimum_grad/Reshape*
_output_shapes	
:�*
T0
�
?gradients/clip_by_value/Minimum_grad/tuple/control_dependency_1Identity.gradients/clip_by_value/Minimum_grad/Reshape_16^gradients/clip_by_value/Minimum_grad/tuple/group_deps*
_output_shapes
: *
T0*A
_class7
53loc:@gradients/clip_by_value/Minimum_grad/Reshape_1
p
gradients/zeros_like	ZerosLike#softmax_cross_entropy_with_logits:1*
T0*
_output_shapes
:	�
�
?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*
T0*
_output_shapes
:	�*

Tdim0
�
4gradients/softmax_cross_entropy_with_logits_grad/mulMul;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims#softmax_cross_entropy_with_logits:1*
T0*
_output_shapes
:	�
�
;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax)softmax_cross_entropy_with_logits/Reshape*
_output_shapes
:	�*
T0
�
4gradients/softmax_cross_entropy_with_logits_grad/NegNeg;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0*
_output_shapes
:	�
�
Agradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:	�
�
6gradients/softmax_cross_entropy_with_logits_grad/mul_1Mul=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_14gradients/softmax_cross_entropy_with_logits_grad/Neg*
_output_shapes
:	�*
T0
�
Agradients/softmax_cross_entropy_with_logits_grad/tuple/group_depsNoOp5^gradients/softmax_cross_entropy_with_logits_grad/mul7^gradients/softmax_cross_entropy_with_logits_grad/mul_1
�
Igradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependencyIdentity4gradients/softmax_cross_entropy_with_logits_grad/mulB^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*G
_class=
;9loc:@gradients/softmax_cross_entropy_with_logits_grad/mul*
_output_shapes
:	�*
T0
�
Kgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1Identity6gradients/softmax_cross_entropy_with_logits_grad/mul_1B^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_with_logits_grad/mul_1*
_output_shapes
:	�
�
gradients/AddN_3AddN/gradients/sub_1_grad/tuple/control_dependency_1gradients/Exp_grad/mul*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1*
N*
_output_shapes
:	�
i
gradients/sub_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�     
k
gradients/sub_grad/Shape_1Const*
valueB"�     *
dtype0*
_output_shapes
:
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/sub_grad/SumSumgradients/AddN_3(gradients/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:	�
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/sub_grad/Sum_1Sumgradients/AddN_3*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
]
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes	
:�*
T0
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
_output_shapes
:	�*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
_output_shapes
:	�*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
_output_shapes
:	�*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
e
gradients/sub_2_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:�
i
gradients/sub_2_grad/Shape_1ShapePlaceholder_4*
T0*
out_type0*
_output_shapes
:
�
*gradients/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_2_grad/Shapegradients/sub_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/sub_2_grad/SumSum=gradients/clip_by_value/Minimum_grad/tuple/control_dependency*gradients/sub_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/sub_2_grad/ReshapeReshapegradients/sub_2_grad/Sumgradients/sub_2_grad/Shape*
_output_shapes	
:�*
T0*
Tshape0
�
gradients/sub_2_grad/Sum_1Sum=gradients/clip_by_value/Minimum_grad/tuple/control_dependency,gradients/sub_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
^
gradients/sub_2_grad/NegNeggradients/sub_2_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_2_grad/Reshape_1Reshapegradients/sub_2_grad/Neggradients/sub_2_grad/Shape_1*#
_output_shapes
:���������*
T0*
Tshape0
m
%gradients/sub_2_grad/tuple/group_depsNoOp^gradients/sub_2_grad/Reshape^gradients/sub_2_grad/Reshape_1
�
-gradients/sub_2_grad/tuple/control_dependencyIdentitygradients/sub_2_grad/Reshape&^gradients/sub_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*/
_class%
#!loc:@gradients/sub_2_grad/Reshape
�
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Reshape_1&^gradients/sub_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_2_grad/Reshape_1*#
_output_shapes
:���������
�
>gradients/softmax_cross_entropy_with_logits/Reshape_grad/ShapeConst*
valueB"�     *
dtype0*
_output_shapes
:
�
@gradients/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapeIgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency>gradients/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
i
gradients/Max_grad/ShapeConst*
valueB"�     *
dtype0*
_output_shapes
:
Y
gradients/Max_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
n
gradients/Max_grad/addAddMax/reduction_indicesgradients/Max_grad/Size*
T0*
_output_shapes
: 
t
gradients/Max_grad/modFloorModgradients/Max_grad/addgradients/Max_grad/Size*
T0*
_output_shapes
: 
]
gradients/Max_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
`
gradients/Max_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
`
gradients/Max_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Max_grad/rangeRangegradients/Max_grad/range/startgradients/Max_grad/Sizegradients/Max_grad/range/delta*
_output_shapes
:*

Tidx0
_
gradients/Max_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Max_grad/FillFillgradients/Max_grad/Shape_1gradients/Max_grad/Fill/value*
T0*

index_type0*
_output_shapes
: 
�
 gradients/Max_grad/DynamicStitchDynamicStitchgradients/Max_grad/rangegradients/Max_grad/modgradients/Max_grad/Shapegradients/Max_grad/Fill*
N*
_output_shapes
:*
T0
�
gradients/Max_grad/ReshapeReshapeMax gradients/Max_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/Max_grad/Reshape_1Reshape-gradients/sub_grad/tuple/control_dependency_1 gradients/Max_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/Max_grad/EqualEqualgradients/Max_grad/Reshapeppo_agent/ppo2_model/pi_3/add*
T0*
_output_shapes
:	�
�
gradients/Max_grad/CastCastgradients/Max_grad/Equal*

SrcT0
*
Truncate( *
_output_shapes
:	�*

DstT0
�
gradients/Max_grad/SumSumgradients/Max_grad/CastMax/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
gradients/Max_grad/Reshape_2Reshapegradients/Max_grad/Sum gradients/Max_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/Max_grad/divRealDivgradients/Max_grad/Castgradients/Max_grad/Reshape_2*
T0*
_output_shapes
:	�
}
gradients/Max_grad/mulMulgradients/Max_grad/divgradients/Max_grad/Reshape_1*
_output_shapes
:	�*
T0
�
gradients/AddN_4AddN-gradients/sub_3_grad/tuple/control_dependency-gradients/sub_2_grad/tuple/control_dependency*
T0*/
_class%
#!loc:@gradients/sub_3_grad/Reshape*
N*
_output_shapes	
:�
�
gradients/AddN_5AddN+gradients/sub_grad/tuple/control_dependency@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshapegradients/Max_grad/mul*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
N*
_output_shapes
:	�
�
2gradients/ppo_agent/ppo2_model/pi_3/add_grad/ShapeConst*
_output_shapes
:*
valueB"�     *
dtype0
~
4gradients/ppo_agent/ppo2_model/pi_3/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Bgradients/ppo_agent/ppo2_model/pi_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/ppo_agent/ppo2_model/pi_3/add_grad/Shape4gradients/ppo_agent/ppo2_model/pi_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0gradients/ppo_agent/ppo2_model/pi_3/add_grad/SumSumgradients/AddN_5Bgradients/ppo_agent/ppo2_model/pi_3/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:	�*
	keep_dims( *

Tidx0
�
4gradients/ppo_agent/ppo2_model/pi_3/add_grad/ReshapeReshape0gradients/ppo_agent/ppo2_model/pi_3/add_grad/Sum2gradients/ppo_agent/ppo2_model/pi_3/add_grad/Shape*
_output_shapes
:	�*
T0*
Tshape0
�
2gradients/ppo_agent/ppo2_model/pi_3/add_grad/Sum_1Sumgradients/AddN_5Dgradients/ppo_agent/ppo2_model/pi_3/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
6gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1Reshape2gradients/ppo_agent/ppo2_model/pi_3/add_grad/Sum_14gradients/ppo_agent/ppo2_model/pi_3/add_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
�
=gradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/group_depsNoOp5^gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape7^gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1
�
Egradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/control_dependencyIdentity4gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape>^gradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape*
_output_shapes
:	�
�
Ggradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/control_dependency_1Identity6gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1>^gradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1*
_output_shapes
:
�
9gradients/ppo_agent/ppo2_model/strided_slice_1_grad/ShapeConst*
valueB"�     *
dtype0*
_output_shapes
:
�
Dgradients/ppo_agent/ppo2_model/strided_slice_1_grad/StridedSliceGradStridedSliceGrad9gradients/ppo_agent/ppo2_model/strided_slice_1_grad/Shape*ppo_agent/ppo2_model/strided_slice_1/stack,ppo_agent/ppo2_model/strided_slice_1/stack_1,ppo_agent/ppo2_model/strided_slice_1/stack_2gradients/AddN_4*
new_axis_mask *
end_mask*
_output_shapes
:	�*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask
�
6gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMulMatMulEgradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/control_dependencyppo_agent/ppo2_model/pi/w/read*
_output_shapes
:	�@*
transpose_a( *
transpose_b(*
T0
�
8gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1MatMul&ppo_agent/ppo2_model/flatten_3/ReshapeEgradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
�
@gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/group_depsNoOp7^gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul9^gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1
�
Hgradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/control_dependencyIdentity6gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMulA^gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/group_deps*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul*
_output_shapes
:	�@*
T0
�
Jgradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/control_dependency_1Identity8gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1A^gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1*
_output_shapes

:@
�
2gradients/ppo_agent/ppo2_model/vf_1/add_grad/ShapeConst*
valueB"�     *
dtype0*
_output_shapes
:
~
4gradients/ppo_agent/ppo2_model/vf_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Bgradients/ppo_agent/ppo2_model/vf_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/ppo_agent/ppo2_model/vf_1/add_grad/Shape4gradients/ppo_agent/ppo2_model/vf_1/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
0gradients/ppo_agent/ppo2_model/vf_1/add_grad/SumSumDgradients/ppo_agent/ppo2_model/strided_slice_1_grad/StridedSliceGradBgradients/ppo_agent/ppo2_model/vf_1/add_grad/BroadcastGradientArgs*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
4gradients/ppo_agent/ppo2_model/vf_1/add_grad/ReshapeReshape0gradients/ppo_agent/ppo2_model/vf_1/add_grad/Sum2gradients/ppo_agent/ppo2_model/vf_1/add_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
�
2gradients/ppo_agent/ppo2_model/vf_1/add_grad/Sum_1SumDgradients/ppo_agent/ppo2_model/strided_slice_1_grad/StridedSliceGradDgradients/ppo_agent/ppo2_model/vf_1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
6gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1Reshape2gradients/ppo_agent/ppo2_model/vf_1/add_grad/Sum_14gradients/ppo_agent/ppo2_model/vf_1/add_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
�
=gradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/group_depsNoOp5^gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape7^gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1
�
Egradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/control_dependencyIdentity4gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape>^gradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape*
_output_shapes
:	�
�
Ggradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/control_dependency_1Identity6gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1>^gradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/group_deps*
_output_shapes
:*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1
�
;gradients/ppo_agent/ppo2_model/flatten_3/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�  @   
�
=gradients/ppo_agent/ppo2_model/flatten_3/Reshape_grad/ReshapeReshapeHgradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/control_dependency;gradients/ppo_agent/ppo2_model/flatten_3/Reshape_grad/Shape*
Tshape0*
_output_shapes
:	�@*
T0
�
6gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMulMatMulEgradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/control_dependencyppo_agent/ppo2_model/vf/w/read*
_output_shapes
:	�@*
transpose_a( *
transpose_b(*
T0
�
8gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1MatMul&ppo_agent/ppo2_model/flatten_2/ReshapeEgradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
�
@gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/group_depsNoOp7^gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul9^gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1
�
Hgradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/control_dependencyIdentity6gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMulA^gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	�@*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul
�
Jgradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/control_dependency_1Identity8gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1A^gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1*
_output_shapes

:@
�
;gradients/ppo_agent/ppo2_model/flatten_2/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�  @   
�
=gradients/ppo_agent/ppo2_model/flatten_2/Reshape_grad/ReshapeReshapeHgradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/control_dependency;gradients/ppo_agent/ppo2_model/flatten_2/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�@
�
gradients/AddN_6AddN=gradients/ppo_agent/ppo2_model/flatten_3/Reshape_grad/Reshape=gradients/ppo_agent/ppo2_model/flatten_2/Reshape_grad/Reshape*
N*
_output_shapes
:	�@*
T0*P
_classF
DBloc:@gradients/ppo_agent/ppo2_model/flatten_3/Reshape_grad/Reshape
�
Hgradients/ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu_grad/LeakyReluGradLeakyReluGradgradients/AddN_6)ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd*
alpha%��L>*
_output_shapes
:	�@*
T0
�
Dgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradHgradients/ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
Igradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/group_depsNoOpE^gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGradI^gradients/ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu_grad/LeakyReluGrad
�
Qgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/control_dependencyIdentityHgradients/ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu_grad/LeakyReluGradJ^gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu_grad/LeakyReluGrad*
_output_shapes
:	�@
�
Sgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/control_dependency_1IdentityDgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGradJ^gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/group_deps*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
�
>gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMulMatMulQgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/control_dependency+ppo_agent/ppo2_model/pi/dense_2/kernel/read*
_output_shapes
:	�@*
transpose_a( *
transpose_b(*
T0
�
@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1MatMul+ppo_agent/ppo2_model/pi_2/dense_1/LeakyReluQgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
�
Hgradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/group_depsNoOp?^gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMulA^gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1
�
Pgradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/control_dependencyIdentity>gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMulI^gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul*
_output_shapes
:	�@
�
Rgradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/control_dependency_1Identity@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1I^gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes

:@@*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1
�
Hgradients/ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu_grad/LeakyReluGradLeakyReluGradPgradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/control_dependency)ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd*
T0*
alpha%��L>*
_output_shapes
:	�@
�
Dgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradHgradients/ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
�
Igradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/group_depsNoOpE^gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGradI^gradients/ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu_grad/LeakyReluGrad
�
Qgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/control_dependencyIdentityHgradients/ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu_grad/LeakyReluGradJ^gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu_grad/LeakyReluGrad*
_output_shapes
:	�@
�
Sgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/control_dependency_1IdentityDgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGradJ^gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGrad
�
>gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMulMatMulQgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/control_dependency+ppo_agent/ppo2_model/pi/dense_1/kernel/read*
T0*
_output_shapes
:	�@*
transpose_a( *
transpose_b(
�
@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1MatMul)ppo_agent/ppo2_model/pi_2/dense/LeakyReluQgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
�
Hgradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/group_depsNoOp?^gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMulA^gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1
�
Pgradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/control_dependencyIdentity>gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMulI^gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul*
_output_shapes
:	�@
�
Rgradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/control_dependency_1Identity@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1I^gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/group_deps*
_output_shapes

:@@*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1
�
Fgradients/ppo_agent/ppo2_model/pi_2/dense/LeakyRelu_grad/LeakyReluGradLeakyReluGradPgradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/control_dependency'ppo_agent/ppo2_model/pi_2/dense/BiasAdd*
T0*
alpha%��L>*
_output_shapes
:	�@
�
Bgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGradBiasAddGradFgradients/ppo_agent/ppo2_model/pi_2/dense/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
Ggradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/group_depsNoOpC^gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGradG^gradients/ppo_agent/ppo2_model/pi_2/dense/LeakyRelu_grad/LeakyReluGrad
�
Ogradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/control_dependencyIdentityFgradients/ppo_agent/ppo2_model/pi_2/dense/LeakyRelu_grad/LeakyReluGradH^gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/LeakyRelu_grad/LeakyReluGrad*
_output_shapes
:	�@
�
Qgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/control_dependency_1IdentityBgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGradH^gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
<gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMulMatMulOgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/control_dependency)ppo_agent/ppo2_model/pi/dense/kernel/read* 
_output_shapes
:
��*
transpose_a( *
transpose_b(*
T0
�
>gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1MatMul)ppo_agent/ppo2_model/pi_2/flatten/ReshapeOgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	�@*
transpose_a(*
transpose_b( *
T0
�
Fgradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/group_depsNoOp=^gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul?^gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1
�
Ngradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/control_dependencyIdentity<gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMulG^gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul* 
_output_shapes
:
��
�
Pgradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/control_dependency_1Identity>gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1G^gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�@
�
>gradients/ppo_agent/ppo2_model/pi_2/flatten/Reshape_grad/ShapeConst*%
valueB"�           *
dtype0*
_output_shapes
:
�
@gradients/ppo_agent/ppo2_model/pi_2/flatten/Reshape_grad/ReshapeReshapeNgradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/control_dependency>gradients/ppo_agent/ppo2_model/pi_2/flatten/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:�
�
Ggradients/ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu_grad/LeakyReluGradLeakyReluGrad@gradients/ppo_agent/ppo2_model/pi_2/flatten/Reshape_grad/Reshape(ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd*
T0*
alpha%��L>*'
_output_shapes
:�
�
Cgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGradBiasAddGradGgradients/ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
Hgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/group_depsNoOpD^gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGradH^gradients/ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu_grad/LeakyReluGrad
�
Pgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/control_dependencyIdentityGgradients/ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu_grad/LeakyReluGradI^gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu_grad/LeakyReluGrad*'
_output_shapes
:�
�
Rgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGradI^gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
=gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/ShapeNShapeN*ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu*ppo_agent/ppo2_model/pi/conv_1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
Jgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/ShapeN*ppo_agent/ppo2_model/pi/conv_1/kernel/readPgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/control_dependency*'
_output_shapes
:�	*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
Kgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter*ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu?gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/ShapeN:1Pgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides

�
Ggradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/group_depsNoOpL^gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilterK^gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropInput
�
Ogradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/control_dependencyIdentityJgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropInputH^gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
:�	
�
Qgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/control_dependency_1IdentityKgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilterH^gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
Ggradients/ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu_grad/LeakyReluGradLeakyReluGradOgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/control_dependency(ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd*
T0*
alpha%��L>*'
_output_shapes
:�	
�
Cgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGradBiasAddGradGgradients/ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
Hgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/group_depsNoOpD^gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGradH^gradients/ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu_grad/LeakyReluGrad
�
Pgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/control_dependencyIdentityGgradients/ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu_grad/LeakyReluGradI^gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu_grad/LeakyReluGrad*'
_output_shapes
:�	
�
Rgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGradI^gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
=gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/ShapeNShapeN0ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu*ppo_agent/ppo2_model/pi/conv_0/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
Jgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/ShapeN*ppo_agent/ppo2_model/pi/conv_0/kernel/readPgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�	*
	dilations
*
T0*
data_formatNHWC*
strides

�
Kgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter0ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu?gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/ShapeN:1Pgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
Ggradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/group_depsNoOpL^gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilterK^gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropInput
�
Ogradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/control_dependencyIdentityJgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropInputH^gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
:�	
�
Qgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/control_dependency_1IdentityKgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilterH^gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
Mgradients/ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu_grad/LeakyReluGradLeakyReluGradOgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/control_dependency.ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd*
T0*
alpha%��L>*'
_output_shapes
:�	
�
Igradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGradBiasAddGradMgradients/ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
Ngradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/group_depsNoOpJ^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGradN^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu_grad/LeakyReluGrad
�
Vgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/control_dependencyIdentityMgradients/ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu_grad/LeakyReluGradO^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu_grad/LeakyReluGrad*'
_output_shapes
:�	
�
Xgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGradO^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
Cgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/ShapeNShapeNppo_agent/ppo2_model/Ob_10ppo_agent/ppo2_model/pi/conv_initial/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
Pgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/ShapeN0ppo_agent/ppo2_model/pi/conv_initial/kernel/readVgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/control_dependency*'
_output_shapes
:�	*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
Qgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterppo_agent/ppo2_model/Ob_1Egradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/ShapeN:1Vgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:
�
Mgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/group_depsNoOpR^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilterQ^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropInput
�
Ugradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/control_dependencyIdentityPgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropInputN^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
:�	
�
Wgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/control_dependency_1IdentityQgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilterN^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*
T0*d
_classZ
XVloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilter
�
global_norm/L2LossL2LossWgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/control_dependency_1*
_output_shapes
: *
T0*d
_classZ
XVloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilter
�
global_norm/L2Loss_1L2LossXgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/control_dependency_1*
T0*\
_classR
PNloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
global_norm/L2Loss_2L2LossQgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/control_dependency_1*
_output_shapes
: *
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilter
�
global_norm/L2Loss_3L2LossRgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/control_dependency_1*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
global_norm/L2Loss_4L2LossQgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/control_dependency_1*
_output_shapes
: *
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilter
�
global_norm/L2Loss_5L2LossRgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/control_dependency_1*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
global_norm/L2Loss_6L2LossPgradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/control_dependency_1*
_output_shapes
: *
T0*Q
_classG
ECloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1
�
global_norm/L2Loss_7L2LossQgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*U
_classK
IGloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
global_norm/L2Loss_8L2LossRgradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_9L2LossSgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
global_norm/L2Loss_10L2LossRgradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_11L2LossSgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
global_norm/L2Loss_12L2LossJgradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_13L2LossGgradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/control_dependency_1*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1*
_output_shapes
: 
�
global_norm/L2Loss_14L2LossJgradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_15L2LossGgradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/control_dependency_1*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1*
_output_shapes
: 
�
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5global_norm/L2Loss_6global_norm/L2Loss_7global_norm/L2Loss_8global_norm/L2Loss_9global_norm/L2Loss_10global_norm/L2Loss_11global_norm/L2Loss_12global_norm/L2Loss_13global_norm/L2Loss_14global_norm/L2Loss_15*
T0*

axis *
N*
_output_shapes
:
[
global_norm/ConstConst*
_output_shapes
:*
valueB: *
dtype0
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
global_norm/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
_output_shapes
: *
T0
Q
global_norm/global_normSqrtglobal_norm/mul*
T0*
_output_shapes
: 
�
VerifyFinite/CheckNumericsCheckNumericsglobal_norm/global_norm*
_output_shapes
: **
messageFound Inf or NaN global norm.*
T0**
_class 
loc:@global_norm/global_norm
�
VerifyFinite/control_dependencyIdentityglobal_norm/global_norm^VerifyFinite/CheckNumerics*
T0**
_class 
loc:@global_norm/global_norm*
_output_shapes
: 
b
clip_by_global_norm/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xVerifyFinite/control_dependency*
T0*
_output_shapes
: 
^
clip_by_global_norm/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
clip_by_global_norm/truediv_1/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 
�
clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
_output_shapes
: *
T0
^
clip_by_global_norm/mul/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
T0*
_output_shapes
: 
�
clip_by_global_norm/mul_1MulWgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/control_dependency_1clip_by_global_norm/mul*&
_output_shapes
:*
T0*d
_classZ
XVloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilter
�
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
T0*d
_classZ
XVloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
clip_by_global_norm/mul_2MulXgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/mul*
_output_shapes
:*
T0*\
_classR
PNloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGrad
�
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
T0*\
_classR
PNloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
clip_by_global_norm/mul_3MulQgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*&
_output_shapes
:*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilter
�
clip_by_global_norm/mul_4MulRgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
clip_by_global_norm/mul_5MulQgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/control_dependency_1clip_by_global_norm/mul*&
_output_shapes
:*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilter
�
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
clip_by_global_norm/mul_6MulRgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
_output_shapes
:*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGrad
�
clip_by_global_norm/mul_7MulPgradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*Q
_classG
ECloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�@
�
*clip_by_global_norm/clip_by_global_norm/_6Identityclip_by_global_norm/mul_7*
T0*Q
_classG
ECloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�@
�
clip_by_global_norm/mul_8MulQgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*U
_classK
IGloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
*clip_by_global_norm/clip_by_global_norm/_7Identityclip_by_global_norm/mul_8*
T0*U
_classK
IGloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
clip_by_global_norm/mul_9MulRgradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
�
*clip_by_global_norm/clip_by_global_norm/_8Identityclip_by_global_norm/mul_9*
_output_shapes

:@@*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1
�
clip_by_global_norm/mul_10MulSgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/mul*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
�
*clip_by_global_norm/clip_by_global_norm/_9Identityclip_by_global_norm/mul_10*
_output_shapes
:@*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGrad
�
clip_by_global_norm/mul_11MulRgradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/mul*
_output_shapes

:@@*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1
�
+clip_by_global_norm/clip_by_global_norm/_10Identityclip_by_global_norm/mul_11*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@@*
T0
�
clip_by_global_norm/mul_12MulSgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
+clip_by_global_norm/clip_by_global_norm/_11Identityclip_by_global_norm/mul_12*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
clip_by_global_norm/mul_13MulJgradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1*
_output_shapes

:@
�
+clip_by_global_norm/clip_by_global_norm/_12Identityclip_by_global_norm/mul_13*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1*
_output_shapes

:@
�
clip_by_global_norm/mul_14MulGgradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/control_dependency_1clip_by_global_norm/mul*
_output_shapes
:*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1
�
+clip_by_global_norm/clip_by_global_norm/_13Identityclip_by_global_norm/mul_14*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1*
_output_shapes
:
�
clip_by_global_norm/mul_15MulJgradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1*
_output_shapes

:@
�
+clip_by_global_norm/clip_by_global_norm/_14Identityclip_by_global_norm/mul_15*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1*
_output_shapes

:@
�
clip_by_global_norm/mul_16MulGgradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/control_dependency_1clip_by_global_norm/mul*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1*
_output_shapes
:*
T0
�
+clip_by_global_norm/clip_by_global_norm/_15Identityclip_by_global_norm/mul_16*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1*
_output_shapes
:
�
beta1_power/initial_valueConst*
valueB
 *fff?*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
x
beta1_power/readIdentitybeta1_power*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: *
T0
�
beta2_power/initial_valueConst*
valueB
 *w�?*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
	container 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
x
beta2_power/readIdentitybeta2_power*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
Rppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
Hppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
valueB
 *    *
dtype0
�
Bppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zerosFillRppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zeros/shape_as_tensorHppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zeros/Const*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*

index_type0*&
_output_shapes
:
�
0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam
VariableV2*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name *>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel
�
7ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/AssignAssign0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamBppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zeros*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
�
5ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/readIdentity0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
Tppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
Jppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zeros/ConstConst*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Dppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zerosFillTppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zeros/shape_as_tensorJppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zeros/Const*&
_output_shapes
:*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*

index_type0
�
2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
	container *
shape:
�
9ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/AssignAssign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1Dppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
7ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/readIdentity2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1*&
_output_shapes
:*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel
�
@ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/Initializer/zerosConst*
_output_shapes
:*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
valueB*    *
dtype0
�
.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias
�
5ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/AssignAssign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam@ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
3ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/readIdentity.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam*
_output_shapes
:*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias
�
Bppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
valueB*    *
dtype0*
_output_shapes
:
�
0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1
VariableV2*
shared_name *<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
7ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/AssignAssign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1Bppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias
�
5ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/readIdentity0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
Lppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zeros/shape_as_tensorConst*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
Bppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zeros/ConstConst*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
<ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zerosFillLppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zeros/shape_as_tensorBppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zeros/Const*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*

index_type0*&
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam
VariableV2*
shared_name *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:
�
1ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/AssignAssign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam<ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zeros*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
/ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/readIdentity*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
Nppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
Dppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
valueB
 *    *
dtype0
�
>ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zerosFillNppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zeros/shape_as_tensorDppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zeros/Const*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*

index_type0*&
_output_shapes
:
�
,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
	container *
shape:
�
3ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/AssignAssign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1>ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zeros*&
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(
�
1ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/readIdentity,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
:ppo_agent/ppo2_model/pi/conv_0/bias/Adam/Initializer/zerosConst*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
valueB*    *
dtype0*
_output_shapes
:
�
(ppo_agent/ppo2_model/pi/conv_0/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
	container 
�
/ppo_agent/ppo2_model/pi/conv_0/bias/Adam/AssignAssign(ppo_agent/ppo2_model/pi/conv_0/bias/Adam:ppo_agent/ppo2_model/pi/conv_0/bias/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
-ppo_agent/ppo2_model/pi/conv_0/bias/Adam/readIdentity(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
<ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
valueB*    *
dtype0*
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
	container *
shape:
�
1ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/AssignAssign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1<ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
/ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/readIdentity*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:*
T0
�
Lppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*%
valueB"            *
dtype0
�
Bppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
valueB
 *    *
dtype0
�
<ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zerosFillLppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zeros/shape_as_tensorBppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zeros/Const*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*

index_type0*&
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam
VariableV2*
shared_name *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:
�
1ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/AssignAssign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam<ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
/ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/readIdentity*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
Nppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
Dppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zeros/ConstConst*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
>ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zerosFillNppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorDppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zeros/Const*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*

index_type0*&
_output_shapes
:
�
,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
	container *
shape:
�
3ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/AssignAssign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1>ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
1ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/readIdentity,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1*&
_output_shapes
:*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel
�
:ppo_agent/ppo2_model/pi/conv_1/bias/Adam/Initializer/zerosConst*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
(ppo_agent/ppo2_model/pi/conv_1/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
	container *
shape:
�
/ppo_agent/ppo2_model/pi/conv_1/bias/Adam/AssignAssign(ppo_agent/ppo2_model/pi/conv_1/bias/Adam:ppo_agent/ppo2_model/pi/conv_1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias
�
-ppo_agent/ppo2_model/pi/conv_1/bias/Adam/readIdentity(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
<ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
	container *
shape:
�
1ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/AssignAssign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1<ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/Initializer/zeros*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
/ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/readIdentity*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
Kppo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
valueB"  @   *
dtype0*
_output_shapes
:
�
Appo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zeros/ConstConst*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;ppo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zerosFillKppo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorAppo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zeros/Const*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*

index_type0*
_output_shapes
:	�@
�
)ppo_agent/ppo2_model/pi/dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	�@*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
	container *
shape:	�@
�
0ppo_agent/ppo2_model/pi/dense/kernel/Adam/AssignAssign)ppo_agent/ppo2_model/pi/dense/kernel/Adam;ppo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
.ppo_agent/ppo2_model/pi/dense/kernel/Adam/readIdentity)ppo_agent/ppo2_model/pi/dense/kernel/Adam*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
Mppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
valueB"  @   *
dtype0*
_output_shapes
:
�
Cppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zeros/ConstConst*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
=ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zerosFillMppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorCppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*

index_type0*
_output_shapes
:	�@
�
+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	�@*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
	container *
shape:	�@
�
2ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/AssignAssign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1=ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zeros*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@*
use_locking(
�
0ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/readIdentity+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@*
T0
�
9ppo_agent/ppo2_model/pi/dense/bias/Adam/Initializer/zerosConst*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
'ppo_agent/ppo2_model/pi/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
	container *
shape:@
�
.ppo_agent/ppo2_model/pi/dense/bias/Adam/AssignAssign'ppo_agent/ppo2_model/pi/dense/bias/Adam9ppo_agent/ppo2_model/pi/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
,ppo_agent/ppo2_model/pi/dense/bias/Adam/readIdentity'ppo_agent/ppo2_model/pi/dense/bias/Adam*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
;ppo_agent/ppo2_model/pi/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:@*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
valueB@*    *
dtype0
�
)ppo_agent/ppo2_model/pi/dense/bias/Adam_1
VariableV2*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
0ppo_agent/ppo2_model/pi/dense/bias/Adam_1/AssignAssign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1;ppo_agent/ppo2_model/pi/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias
�
.ppo_agent/ppo2_model/pi/dense/bias/Adam_1/readIdentity)ppo_agent/ppo2_model/pi/dense/bias/Adam_1*
_output_shapes
:@*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias
�
Mppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
�
Cppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
valueB
 *    
�
=ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zerosFillMppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorCppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*

index_type0*
_output_shapes

:@@
�
+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
	container *
shape
:@@
�
2ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/AssignAssign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam=ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
0ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/readIdentity+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@*
T0
�
Oppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
�
Eppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
?ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zerosFillOppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorEppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*

index_type0*
_output_shapes

:@@
�
-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1
VariableV2*
shared_name *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
�
4ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/AssignAssign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1?ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zeros*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
2ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/readIdentity-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1*
_output_shapes

:@@*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel
�
;ppo_agent/ppo2_model/pi/dense_1/bias/Adam/Initializer/zerosConst*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
)ppo_agent/ppo2_model/pi/dense_1/bias/Adam
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias
�
0ppo_agent/ppo2_model/pi/dense_1/bias/Adam/AssignAssign)ppo_agent/ppo2_model/pi/dense_1/bias/Adam;ppo_agent/ppo2_model/pi/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
.ppo_agent/ppo2_model/pi/dense_1/bias/Adam/readIdentity)ppo_agent/ppo2_model/pi/dense_1/bias/Adam*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@*
T0
�
=ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/Initializer/zerosConst*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
	container *
shape:@
�
2ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/AssignAssign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1=ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/Initializer/zeros*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
0ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/readIdentity+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@*
T0
�
Mppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
�
Cppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zeros/ConstConst*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
=ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zerosFillMppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorCppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zeros/Const*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*

index_type0*
_output_shapes

:@@
�
+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam
VariableV2*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name 
�
2ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/AssignAssign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam=ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel
�
0ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/readIdentity+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam*
_output_shapes

:@@*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel
�
Oppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
valueB"@   @   *
dtype0
�
Eppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
valueB
 *    
�
?ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zerosFillOppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorEppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zeros/Const*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*

index_type0*
_output_shapes

:@@
�
-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
	container *
shape
:@@
�
4ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/AssignAssign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1?ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
2ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/readIdentity-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
;ppo_agent/ppo2_model/pi/dense_2/bias/Adam/Initializer/zerosConst*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
)ppo_agent/ppo2_model/pi/dense_2/bias/Adam
VariableV2*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
0ppo_agent/ppo2_model/pi/dense_2/bias/Adam/AssignAssign)ppo_agent/ppo2_model/pi/dense_2/bias/Adam;ppo_agent/ppo2_model/pi/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
.ppo_agent/ppo2_model/pi/dense_2/bias/Adam/readIdentity)ppo_agent/ppo2_model/pi/dense_2/bias/Adam*
_output_shapes
:@*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias
�
=ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/Initializer/zerosConst*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
	container *
shape:@
�
2ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/AssignAssign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1=ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/Initializer/zeros*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
0ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/readIdentity+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
0ppo_agent/ppo2_model/pi/w/Adam/Initializer/zerosConst*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
ppo_agent/ppo2_model/pi/w/Adam
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
	container 
�
%ppo_agent/ppo2_model/pi/w/Adam/AssignAssignppo_agent/ppo2_model/pi/w/Adam0ppo_agent/ppo2_model/pi/w/Adam/Initializer/zeros*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
#ppo_agent/ppo2_model/pi/w/Adam/readIdentityppo_agent/ppo2_model/pi/w/Adam*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@*
T0
�
2ppo_agent/ppo2_model/pi/w/Adam_1/Initializer/zerosConst*
_output_shapes

:@*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
valueB@*    *
dtype0
�
 ppo_agent/ppo2_model/pi/w/Adam_1
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/pi/w
�
'ppo_agent/ppo2_model/pi/w/Adam_1/AssignAssign ppo_agent/ppo2_model/pi/w/Adam_12ppo_agent/ppo2_model/pi/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w
�
%ppo_agent/ppo2_model/pi/w/Adam_1/readIdentity ppo_agent/ppo2_model/pi/w/Adam_1*
_output_shapes

:@*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w
�
0ppo_agent/ppo2_model/pi/b/Adam/Initializer/zerosConst*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
valueB*    *
dtype0*
_output_shapes
:
�
ppo_agent/ppo2_model/pi/b/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
	container *
shape:
�
%ppo_agent/ppo2_model/pi/b/Adam/AssignAssignppo_agent/ppo2_model/pi/b/Adam0ppo_agent/ppo2_model/pi/b/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(
�
#ppo_agent/ppo2_model/pi/b/Adam/readIdentityppo_agent/ppo2_model/pi/b/Adam*
_output_shapes
:*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b
�
2ppo_agent/ppo2_model/pi/b/Adam_1/Initializer/zerosConst*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
valueB*    *
dtype0*
_output_shapes
:
�
 ppo_agent/ppo2_model/pi/b/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
	container 
�
'ppo_agent/ppo2_model/pi/b/Adam_1/AssignAssign ppo_agent/ppo2_model/pi/b/Adam_12ppo_agent/ppo2_model/pi/b/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
%ppo_agent/ppo2_model/pi/b/Adam_1/readIdentity ppo_agent/ppo2_model/pi/b/Adam_1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
0ppo_agent/ppo2_model/vf/w/Adam/Initializer/zerosConst*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
ppo_agent/ppo2_model/vf/w/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
	container *
shape
:@
�
%ppo_agent/ppo2_model/vf/w/Adam/AssignAssignppo_agent/ppo2_model/vf/w/Adam0ppo_agent/ppo2_model/vf/w/Adam/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(
�
#ppo_agent/ppo2_model/vf/w/Adam/readIdentityppo_agent/ppo2_model/vf/w/Adam*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
2ppo_agent/ppo2_model/vf/w/Adam_1/Initializer/zerosConst*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
 ppo_agent/ppo2_model/vf/w/Adam_1
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
	container 
�
'ppo_agent/ppo2_model/vf/w/Adam_1/AssignAssign ppo_agent/ppo2_model/vf/w/Adam_12ppo_agent/ppo2_model/vf/w/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
%ppo_agent/ppo2_model/vf/w/Adam_1/readIdentity ppo_agent/ppo2_model/vf/w/Adam_1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
0ppo_agent/ppo2_model/vf/b/Adam/Initializer/zerosConst*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
valueB*    *
dtype0*
_output_shapes
:
�
ppo_agent/ppo2_model/vf/b/Adam
VariableV2*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
%ppo_agent/ppo2_model/vf/b/Adam/AssignAssignppo_agent/ppo2_model/vf/b/Adam0ppo_agent/ppo2_model/vf/b/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(
�
#ppo_agent/ppo2_model/vf/b/Adam/readIdentityppo_agent/ppo2_model/vf/b/Adam*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
2ppo_agent/ppo2_model/vf/b/Adam_1/Initializer/zerosConst*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
valueB*    *
dtype0*
_output_shapes
:
�
 ppo_agent/ppo2_model/vf/b/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
'ppo_agent/ppo2_model/vf/b/Adam_1/AssignAssign ppo_agent/ppo2_model/vf/b/Adam_12ppo_agent/ppo2_model/vf/b/Adam_1/Initializer/zeros*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
%ppo_agent/ppo2_model/vf/b/Adam_1/readIdentity ppo_agent/ppo2_model/vf/b/Adam_1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *��'7*
dtype0
�
AAdam/update_ppo_agent/ppo2_model/pi/conv_initial/kernel/ApplyAdam	ApplyAdam+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
use_nesterov( *&
_output_shapes
:*
use_locking( *
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel
�
?Adam/update_ppo_agent/ppo2_model/pi/conv_initial/bias/ApplyAdam	ApplyAdam)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_locking( *
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
use_nesterov( *
_output_shapes
:
�
;Adam/update_ppo_agent/ppo2_model/pi/conv_0/kernel/ApplyAdam	ApplyAdam%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_locking( *
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
use_nesterov( *&
_output_shapes
:
�
9Adam/update_ppo_agent/ppo2_model/pi/conv_0/bias/ApplyAdam	ApplyAdam#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias
�
;Adam/update_ppo_agent/ppo2_model/pi/conv_1/kernel/ApplyAdam	ApplyAdam%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4*
use_locking( *
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
use_nesterov( *&
_output_shapes
:
�
9Adam/update_ppo_agent/ppo2_model/pi/conv_1/bias/ApplyAdam	ApplyAdam#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_5*
use_locking( *
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
use_nesterov( *
_output_shapes
:
�
:Adam/update_ppo_agent/ppo2_model/pi/dense/kernel/ApplyAdam	ApplyAdam$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_6*
use_locking( *
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
use_nesterov( *
_output_shapes
:	�@
�
8Adam/update_ppo_agent/ppo2_model/pi/dense/bias/ApplyAdam	ApplyAdam"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_7*
use_locking( *
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
use_nesterov( *
_output_shapes
:@
�
<Adam/update_ppo_agent/ppo2_model/pi/dense_1/kernel/ApplyAdam	ApplyAdam&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_8*
use_locking( *
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
use_nesterov( *
_output_shapes

:@@
�
:Adam/update_ppo_agent/ppo2_model/pi/dense_1/bias/ApplyAdam	ApplyAdam$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_9*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0
�
<Adam/update_ppo_agent/ppo2_model/pi/dense_2/kernel/ApplyAdam	ApplyAdam&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_10*
use_locking( *
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
use_nesterov( *
_output_shapes

:@@
�
:Adam/update_ppo_agent/ppo2_model/pi/dense_2/bias/ApplyAdam	ApplyAdam$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_11*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0
�
/Adam/update_ppo_agent/ppo2_model/pi/w/ApplyAdam	ApplyAdamppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_12*
use_locking( *
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
use_nesterov( *
_output_shapes

:@
�
/Adam/update_ppo_agent/ppo2_model/pi/b/ApplyAdam	ApplyAdamppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_13*
use_locking( *
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
use_nesterov( *
_output_shapes
:
�
/Adam/update_ppo_agent/ppo2_model/vf/w/ApplyAdam	ApplyAdamppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_14*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0
�
/Adam/update_ppo_agent/ppo2_model/vf/b/ApplyAdam	ApplyAdamppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_15*
_output_shapes
:*
use_locking( *
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
use_nesterov( 
�
Adam/mulMulbeta1_power/read
Adam/beta10^Adam/update_ppo_agent/ppo2_model/pi/b/ApplyAdam:^Adam/update_ppo_agent/ppo2_model/pi/conv_0/bias/ApplyAdam<^Adam/update_ppo_agent/ppo2_model/pi/conv_0/kernel/ApplyAdam:^Adam/update_ppo_agent/ppo2_model/pi/conv_1/bias/ApplyAdam<^Adam/update_ppo_agent/ppo2_model/pi/conv_1/kernel/ApplyAdam@^Adam/update_ppo_agent/ppo2_model/pi/conv_initial/bias/ApplyAdamB^Adam/update_ppo_agent/ppo2_model/pi/conv_initial/kernel/ApplyAdam9^Adam/update_ppo_agent/ppo2_model/pi/dense/bias/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense/kernel/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense_1/bias/ApplyAdam=^Adam/update_ppo_agent/ppo2_model/pi/dense_1/kernel/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense_2/bias/ApplyAdam=^Adam/update_ppo_agent/ppo2_model/pi/dense_2/kernel/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/pi/w/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/vf/b/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/vf/w/ApplyAdam*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�

Adam/mul_1Mulbeta2_power/read
Adam/beta20^Adam/update_ppo_agent/ppo2_model/pi/b/ApplyAdam:^Adam/update_ppo_agent/ppo2_model/pi/conv_0/bias/ApplyAdam<^Adam/update_ppo_agent/ppo2_model/pi/conv_0/kernel/ApplyAdam:^Adam/update_ppo_agent/ppo2_model/pi/conv_1/bias/ApplyAdam<^Adam/update_ppo_agent/ppo2_model/pi/conv_1/kernel/ApplyAdam@^Adam/update_ppo_agent/ppo2_model/pi/conv_initial/bias/ApplyAdamB^Adam/update_ppo_agent/ppo2_model/pi/conv_initial/kernel/ApplyAdam9^Adam/update_ppo_agent/ppo2_model/pi/dense/bias/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense/kernel/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense_1/bias/ApplyAdam=^Adam/update_ppo_agent/ppo2_model/pi/dense_1/kernel/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense_2/bias/ApplyAdam=^Adam/update_ppo_agent/ppo2_model/pi/dense_2/kernel/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/pi/w/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/vf/b/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/vf/w/ApplyAdam*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
AdamNoOp^Adam/Assign^Adam/Assign_10^Adam/update_ppo_agent/ppo2_model/pi/b/ApplyAdam:^Adam/update_ppo_agent/ppo2_model/pi/conv_0/bias/ApplyAdam<^Adam/update_ppo_agent/ppo2_model/pi/conv_0/kernel/ApplyAdam:^Adam/update_ppo_agent/ppo2_model/pi/conv_1/bias/ApplyAdam<^Adam/update_ppo_agent/ppo2_model/pi/conv_1/kernel/ApplyAdam@^Adam/update_ppo_agent/ppo2_model/pi/conv_initial/bias/ApplyAdamB^Adam/update_ppo_agent/ppo2_model/pi/conv_initial/kernel/ApplyAdam9^Adam/update_ppo_agent/ppo2_model/pi/dense/bias/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense/kernel/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense_1/bias/ApplyAdam=^Adam/update_ppo_agent/ppo2_model/pi/dense_1/kernel/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense_2/bias/ApplyAdam=^Adam/update_ppo_agent/ppo2_model/pi/dense_2/kernel/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/pi/w/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/vf/b/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/vf/w/ApplyAdam
�
initNoOp^beta1_power/Assign^beta2_power/Assign&^ppo_agent/ppo2_model/pi/b/Adam/Assign(^ppo_agent/ppo2_model/pi/b/Adam_1/Assign!^ppo_agent/ppo2_model/pi/b/Assign0^ppo_agent/ppo2_model/pi/conv_0/bias/Adam/Assign2^ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/Assign+^ppo_agent/ppo2_model/pi/conv_0/bias/Assign2^ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Assign4^ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Assign-^ppo_agent/ppo2_model/pi/conv_0/kernel/Assign0^ppo_agent/ppo2_model/pi/conv_1/bias/Adam/Assign2^ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/Assign+^ppo_agent/ppo2_model/pi/conv_1/bias/Assign2^ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Assign4^ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Assign-^ppo_agent/ppo2_model/pi/conv_1/kernel/Assign6^ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/Assign8^ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/Assign1^ppo_agent/ppo2_model/pi/conv_initial/bias/Assign8^ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Assign:^ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Assign3^ppo_agent/ppo2_model/pi/conv_initial/kernel/Assign/^ppo_agent/ppo2_model/pi/dense/bias/Adam/Assign1^ppo_agent/ppo2_model/pi/dense/bias/Adam_1/Assign*^ppo_agent/ppo2_model/pi/dense/bias/Assign1^ppo_agent/ppo2_model/pi/dense/kernel/Adam/Assign3^ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Assign,^ppo_agent/ppo2_model/pi/dense/kernel/Assign1^ppo_agent/ppo2_model/pi/dense_1/bias/Adam/Assign3^ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/Assign,^ppo_agent/ppo2_model/pi/dense_1/bias/Assign3^ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Assign5^ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Assign.^ppo_agent/ppo2_model/pi/dense_1/kernel/Assign1^ppo_agent/ppo2_model/pi/dense_2/bias/Adam/Assign3^ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/Assign,^ppo_agent/ppo2_model/pi/dense_2/bias/Assign3^ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Assign5^ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Assign.^ppo_agent/ppo2_model/pi/dense_2/kernel/Assign&^ppo_agent/ppo2_model/pi/w/Adam/Assign(^ppo_agent/ppo2_model/pi/w/Adam_1/Assign!^ppo_agent/ppo2_model/pi/w/Assign&^ppo_agent/ppo2_model/vf/b/Adam/Assign(^ppo_agent/ppo2_model/vf/b/Adam_1/Assign!^ppo_agent/ppo2_model/vf/b/Assign&^ppo_agent/ppo2_model/vf/w/Adam/Assign(^ppo_agent/ppo2_model/vf/w/Adam_1/Assign!^ppo_agent/ppo2_model/vf/w/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_46de323a738e485287a712cb6a761e4a/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
�
save/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:2*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422
�
save/AssignAssignbeta1_powersave/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_2Assignppo_agent/ppo2_model/pi/bsave/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save/RestoreV2:4*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(
�
save/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave/RestoreV2:5*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save/RestoreV2:7*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave/RestoreV2:8*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel
�
save/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave/RestoreV2:9*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
�
save/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save/RestoreV2:10*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel
�
save/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave/RestoreV2:12*
_output_shapes
:*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(
�
save/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave/RestoreV2:15*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
save/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save/RestoreV2:16*&
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(
�
save/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias
�
save/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave/RestoreV2:18*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(
�
save/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save/RestoreV2:19*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave/RestoreV2:20*&
_output_shapes
:*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(
�
save/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave/RestoreV2:21*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
save/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save/RestoreV2:22*&
_output_shapes
:*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(
�
save/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave/RestoreV2:23*
_output_shapes
:@*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(
�
save/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave/RestoreV2:24*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save/RestoreV2:25*
_output_shapes
:@*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(
�
save/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave/RestoreV2:30*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave/RestoreV2:32*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel
�
save/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save/RestoreV2:34*
_output_shapes

:@@*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(
�
save/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave/RestoreV2:36*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave/RestoreV2:39*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save/Assign_41Assignppo_agent/ppo2_model/pi/wsave/RestoreV2:41*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w
�
save/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave/RestoreV2:42*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save/Assign_44Assignppo_agent/ppo2_model/vf/bsave/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave/RestoreV2:45*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save/Assign_47Assignppo_agent/ppo2_model/vf/wsave/RestoreV2:47*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w
�
save/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave/RestoreV2:48*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save/RestoreV2:49*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_c6a8e9aceabf4b2580a3bf0ee7e99f7a/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
�
save_1/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_1/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
�
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
�
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
�
save_1/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:2*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422
�
save_1/AssignAssignbeta1_powersave_1/RestoreV2*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save_1/Assign_1Assignbeta2_powersave_1/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_2Assignppo_agent/ppo2_model/pi/bsave_1/RestoreV2:2*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_1/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_1/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_1/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_1/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_1/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_1/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_1/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_1/RestoreV2:7*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_1/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_1/RestoreV2:8*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
save_1/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_1/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_1/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_1/RestoreV2:11*
_output_shapes
:*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(
�
save_1/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_1/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_1/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_1/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_1/RestoreV2:15*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
�
save_1/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_1/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_1/RestoreV2:17*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_1/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_1/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_1/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_1/RestoreV2:20*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
save_1/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_1/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_1/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
�
save_1/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_1/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_1/RestoreV2:24*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias
�
save_1/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_1/RestoreV2:25*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_1/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_1/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_1/RestoreV2:27*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@*
use_locking(
�
save_1/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_1/RestoreV2:28*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@*
use_locking(
�
save_1/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_1/RestoreV2:29*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(
�
save_1/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_1/RestoreV2:30*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_1/RestoreV2:31*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(
�
save_1/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_1/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_1/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_1/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_1/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_1/RestoreV2:34*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
save_1/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_1/RestoreV2:35*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_1/RestoreV2:36*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias
�
save_1/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_1/RestoreV2:37*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias
�
save_1/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_1/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_1/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_1/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_1/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_1/RestoreV2:40*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel
�
save_1/Assign_41Assignppo_agent/ppo2_model/pi/wsave_1/RestoreV2:41*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_1/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_1/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_1/RestoreV2:43*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_1/Assign_44Assignppo_agent/ppo2_model/vf/bsave_1/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_1/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_1/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_1/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_1/RestoreV2:46*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(
�
save_1/Assign_47Assignppo_agent/ppo2_model/vf/wsave_1/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_1/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_1/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
_output_shapes
: *
shape: *
dtype0
�
save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_8606144742e542a6a8249d0f81547580/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_2/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
�
save_2/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_2/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_2/ShardedFilename
�
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
�
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
_output_shapes
: *
T0
�
save_2/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_2/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b
�
save_2/Assign_1Assignbeta2_powersave_2/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b
�
save_2/Assign_2Assignppo_agent/ppo2_model/pi/bsave_2/RestoreV2:2*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(
�
save_2/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_2/RestoreV2:3*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_2/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_2/RestoreV2:4*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_2/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_2/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_2/RestoreV2:6*
_output_shapes
:*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(
�
save_2/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_2/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_2/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_2/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_2/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_2/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_2/RestoreV2:10*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
�
save_2/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_2/RestoreV2:11*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_2/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_2/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_2/RestoreV2:13*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_2/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_2/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_2/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_2/RestoreV2:15*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
save_2/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_2/RestoreV2:16*&
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(
�
save_2/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_2/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias
�
save_2/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_2/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_2/RestoreV2:19*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(
�
save_2/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_2/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_2/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_2/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_2/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_2/RestoreV2:22*&
_output_shapes
:*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(
�
save_2/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_2/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_2/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_2/RestoreV2:25*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_2/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_2/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_2/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_2/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_2/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_2/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_2/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_2/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_2/RestoreV2:30*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_2/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_2/RestoreV2:31*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_2/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_2/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_2/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_2/RestoreV2:33*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel
�
save_2/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_2/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_2/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_2/RestoreV2:35*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(
�
save_2/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_2/RestoreV2:36*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_2/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_2/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_2/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_2/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_2/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_2/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_2/Assign_41Assignppo_agent/ppo2_model/pi/wsave_2/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_2/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_2/RestoreV2:43*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w
�
save_2/Assign_44Assignppo_agent/ppo2_model/vf/bsave_2/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_2/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_2/RestoreV2:45*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b
�
save_2/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_2/RestoreV2:46*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b
�
save_2/Assign_47Assignppo_agent/ppo2_model/vf/wsave_2/RestoreV2:47*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save_2/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_2/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_2/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_43^save_2/Assign_44^save_2/Assign_45^save_2/Assign_46^save_2/Assign_47^save_2/Assign_48^save_2/Assign_49^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_d86e34c53d684baca243d8a11e2e55a9/part*
dtype0*
_output_shapes
: 
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_3/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_3/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
�
save_3/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_3/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: 
�
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
�
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
�
save_3/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_3/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422
�
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b
�
save_3/Assign_1Assignbeta2_powersave_3/RestoreV2:1*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(
�
save_3/Assign_2Assignppo_agent/ppo2_model/pi/bsave_3/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_3/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_3/RestoreV2:3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b
�
save_3/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_3/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_3/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_3/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_3/RestoreV2:6*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias
�
save_3/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_3/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_3/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_3/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_3/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_3/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_3/RestoreV2:10*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
save_3/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_3/RestoreV2:11*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_3/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_3/RestoreV2:12*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias
�
save_3/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_3/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_3/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_3/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_3/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_3/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_3/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_3/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_3/RestoreV2:17*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_3/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_3/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_3/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_3/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_3/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_3/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_3/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_3/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_3/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_3/RestoreV2:23*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_3/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_3/RestoreV2:24*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_3/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_3/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_3/RestoreV2:26*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@*
use_locking(
�
save_3/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_3/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_3/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_3/RestoreV2:28*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@*
use_locking(*
T0
�
save_3/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_3/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_3/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_3/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_3/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_3/RestoreV2:33*
_output_shapes

:@@*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(
�
save_3/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_3/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_3/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_3/RestoreV2:36*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_3/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_3/RestoreV2:38*
_output_shapes

:@@*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(
�
save_3/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_3/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_3/RestoreV2:40*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel
�
save_3/Assign_41Assignppo_agent/ppo2_model/pi/wsave_3/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_3/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_3/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_44Assignppo_agent/ppo2_model/vf/bsave_3/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_3/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_3/RestoreV2:45*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b
�
save_3/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_3/RestoreV2:46*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b
�
save_3/Assign_47Assignppo_agent/ppo2_model/vf/wsave_3/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_3/RestoreV2:48*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w
�
save_3/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_3/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_4/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_3c4a96386534467ab0f5e45260fa6581/part*
dtype0
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_4/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_4/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
�
save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
�
save_4/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_4/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_4/ShardedFilename
�
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(
�
save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
_output_shapes
: *
T0
�
save_4/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_4/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422
�
save_4/AssignAssignbeta1_powersave_4/RestoreV2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_4/Assign_1Assignbeta2_powersave_4/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_4/Assign_2Assignppo_agent/ppo2_model/pi/bsave_4/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_4/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_4/RestoreV2:3*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_4/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_4/RestoreV2:4*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_4/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_4/RestoreV2:5*
_output_shapes
:*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(
�
save_4/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_4/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_4/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_4/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_4/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_4/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_4/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_4/RestoreV2:9*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel
�
save_4/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_4/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_4/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_4/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_4/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_4/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_4/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_4/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_4/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_4/RestoreV2:14*&
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(
�
save_4/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_4/RestoreV2:15*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel
�
save_4/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_4/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_4/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_4/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_4/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_4/RestoreV2:18*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_4/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_4/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_4/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_4/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_4/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_4/RestoreV2:21*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
save_4/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_4/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
�
save_4/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_4/RestoreV2:23*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias
�
save_4/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_4/RestoreV2:24*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_4/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_4/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_4/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_4/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_4/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_4/RestoreV2:27*
validate_shape(*
_output_shapes
:	�@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel
�
save_4/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_4/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_4/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_4/RestoreV2:29*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_4/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_4/RestoreV2:30*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_4/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_4/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_4/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_4/RestoreV2:32*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel
�
save_4/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_4/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_4/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_4/RestoreV2:34*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_4/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_4/RestoreV2:35*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(
�
save_4/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_4/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_4/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_4/RestoreV2:37*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_4/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_4/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_4/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_4/RestoreV2:39*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
save_4/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_4/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_4/Assign_41Assignppo_agent/ppo2_model/pi/wsave_4/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_4/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_4/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_4/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_4/RestoreV2:43*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w
�
save_4/Assign_44Assignppo_agent/ppo2_model/vf/bsave_4/RestoreV2:44*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_4/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_4/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_4/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_4/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_4/Assign_47Assignppo_agent/ppo2_model/vf/wsave_4/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_4/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_4/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_4/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_4/RestoreV2:49*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(
�
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_40^save_4/Assign_41^save_4/Assign_42^save_4/Assign_43^save_4/Assign_44^save_4/Assign_45^save_4/Assign_46^save_4/Assign_47^save_4/Assign_48^save_4/Assign_49^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_5/StringJoin/inputs_1Const*<
value3B1 B+_temp_6e85a885efb9462da4ee97baf7e4cbc8/part*
dtype0*
_output_shapes
: 
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_5/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_5/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
�
save_5/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_5/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
T0*)
_class
loc:@save_5/ShardedFilename*
_output_shapes
: 
�
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
N*
_output_shapes
:*
T0*

axis 
�
save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(
�
save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
�
save_5/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_5/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:2*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422
�
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_5/Assign_1Assignbeta2_powersave_5/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_5/Assign_2Assignppo_agent/ppo2_model/pi/bsave_5/RestoreV2:2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b
�
save_5/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_5/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_5/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_5/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_5/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_5/RestoreV2:5*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_5/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_5/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_5/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_5/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_5/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_5/RestoreV2:8*&
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(
�
save_5/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_5/RestoreV2:9*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel
�
save_5/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_5/RestoreV2:10*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
�
save_5/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_5/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_5/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_5/RestoreV2:12*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_5/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_5/RestoreV2:13*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_5/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_5/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_5/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_5/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_5/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_5/RestoreV2:16*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
�
save_5/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_5/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_5/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_5/RestoreV2:18*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(
�
save_5/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_5/RestoreV2:19*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias
�
save_5/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_5/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_5/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_5/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_5/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_5/RestoreV2:22*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel
�
save_5/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_5/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_5/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_5/RestoreV2:24*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias
�
save_5/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_5/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_5/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_5/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_5/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_5/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_5/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_5/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_5/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_5/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_5/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_5/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_5/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_5/RestoreV2:31*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_5/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_5/RestoreV2:32*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
save_5/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_5/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_5/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_5/RestoreV2:34*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel
�
save_5/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_5/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_5/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_5/RestoreV2:36*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_5/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_5/RestoreV2:37*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_5/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_5/RestoreV2:38*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_5/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_5/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_5/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_5/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_5/Assign_41Assignppo_agent/ppo2_model/pi/wsave_5/RestoreV2:41*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w
�
save_5/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_5/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_5/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_5/RestoreV2:43*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w
�
save_5/Assign_44Assignppo_agent/ppo2_model/vf/bsave_5/RestoreV2:44*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(
�
save_5/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_5/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_5/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_5/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_5/Assign_47Assignppo_agent/ppo2_model/vf/wsave_5/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_5/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_5/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_5/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_5/RestoreV2:49*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(
�
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_40^save_5/Assign_41^save_5/Assign_42^save_5/Assign_43^save_5/Assign_44^save_5/Assign_45^save_5/Assign_46^save_5/Assign_47^save_5/Assign_48^save_5/Assign_49^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_6/StringJoin/inputs_1Const*<
value3B1 B+_temp_dc071443197c4d40bc8d62b7cc5fcea2/part*
dtype0*
_output_shapes
: 
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_6/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
�
save_6/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_6/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
T0*)
_class
loc:@save_6/ShardedFilename*
_output_shapes
: 
�
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(
�
save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
T0*
_output_shapes
: 
�
save_6/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:2*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1
�
!save_6/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422
�
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_6/Assign_1Assignbeta2_powersave_6/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_6/Assign_2Assignppo_agent/ppo2_model/pi/bsave_6/RestoreV2:2*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_6/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_6/RestoreV2:3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b
�
save_6/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_6/RestoreV2:4*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_6/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_6/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_6/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_6/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_6/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_6/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_6/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_6/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_6/RestoreV2:10*&
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(
�
save_6/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_6/RestoreV2:11*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_6/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_6/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_6/RestoreV2:13*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_6/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_6/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_6/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_6/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_6/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_6/RestoreV2:16*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
�
save_6/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_6/RestoreV2:17*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_6/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_6/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias
�
save_6/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_6/RestoreV2:19*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias
�
save_6/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_6/RestoreV2:20*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
�
save_6/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_6/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_6/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_6/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_6/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_6/RestoreV2:23*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_6/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_6/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_6/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_6/RestoreV2:26*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@*
use_locking(*
T0
�
save_6/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_6/RestoreV2:27*
_output_shapes
:	�@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(
�
save_6/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_6/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_6/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_6/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_6/RestoreV2:30*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(
�
save_6/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_6/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_6/RestoreV2:32*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_6/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_6/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_6/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_6/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_6/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_6/RestoreV2:35*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(
�
save_6/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_6/RestoreV2:36*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_6/RestoreV2:37*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias
�
save_6/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_6/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_6/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_6/RestoreV2:39*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel
�
save_6/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_6/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_6/Assign_41Assignppo_agent/ppo2_model/pi/wsave_6/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_6/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_6/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_6/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_6/RestoreV2:43*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_6/Assign_44Assignppo_agent/ppo2_model/vf/bsave_6/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_6/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_6/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_6/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_6/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_6/Assign_47Assignppo_agent/ppo2_model/vf/wsave_6/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_6/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_6/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_6/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_6/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_40^save_6/Assign_41^save_6/Assign_42^save_6/Assign_43^save_6/Assign_44^save_6/Assign_45^save_6/Assign_46^save_6/Assign_47^save_6/Assign_48^save_6/Assign_49^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
_output_shapes
: *
shape: *
dtype0
�
save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_990e0dfaf16c4a6782c3bb4b69083a15/part*
dtype0*
_output_shapes
: 
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_7/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_7/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
�
save_7/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_7/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_7/ShardedFilename
�
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*
N*
_output_shapes
:*
T0*

axis 
�
save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(
�
save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
T0*
_output_shapes
: 
�
save_7/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_7/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_7/AssignAssignbeta1_powersave_7/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_7/Assign_1Assignbeta2_powersave_7/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_7/Assign_2Assignppo_agent/ppo2_model/pi/bsave_7/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_7/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_7/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_7/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_7/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_7/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_7/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_7/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_7/RestoreV2:6*
_output_shapes
:*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(
�
save_7/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_7/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_7/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_7/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_7/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_7/RestoreV2:9*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
save_7/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_7/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_7/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_7/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_7/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_7/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_7/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_7/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_7/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_7/RestoreV2:14*&
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(
�
save_7/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_7/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_7/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_7/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_7/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_7/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias
�
save_7/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_7/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_7/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_7/RestoreV2:19*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_7/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_7/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_7/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_7/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_7/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_7/RestoreV2:22*&
_output_shapes
:*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(
�
save_7/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_7/RestoreV2:23*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_7/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_7/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_7/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_7/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_7/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_7/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_7/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_7/RestoreV2:27*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@*
use_locking(
�
save_7/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_7/RestoreV2:28*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@*
use_locking(*
T0
�
save_7/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_7/RestoreV2:29*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(
�
save_7/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_7/RestoreV2:30*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(
�
save_7/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_7/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_7/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_7/RestoreV2:32*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
save_7/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_7/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_7/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_7/RestoreV2:34*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel
�
save_7/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_7/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_7/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_7/RestoreV2:36*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias
�
save_7/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_7/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_7/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_7/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_7/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_7/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_7/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_7/RestoreV2:40*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_7/Assign_41Assignppo_agent/ppo2_model/pi/wsave_7/RestoreV2:41*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save_7/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_7/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_7/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_7/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_7/Assign_44Assignppo_agent/ppo2_model/vf/bsave_7/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_7/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_7/RestoreV2:45*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_7/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_7/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_7/Assign_47Assignppo_agent/ppo2_model/vf/wsave_7/RestoreV2:47*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(
�
save_7/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_7/RestoreV2:48*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w
�
save_7/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_7/RestoreV2:49*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w
�
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_40^save_7/Assign_41^save_7/Assign_42^save_7/Assign_43^save_7/Assign_44^save_7/Assign_45^save_7/Assign_46^save_7/Assign_47^save_7/Assign_48^save_7/Assign_49^save_7/Assign_5^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard "B
save_7/Const:0save_7/Identity:0save_7/restore_all (5 @F8"�
trainable_variables��
�
-ppo_agent/ppo2_model/pi/conv_initial/kernel:02ppo_agent/ppo2_model/pi/conv_initial/kernel/Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/read:02Hppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform:08
�
+ppo_agent/ppo2_model/pi/conv_initial/bias:00ppo_agent/ppo2_model/pi/conv_initial/bias/Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/read:02=ppo_agent/ppo2_model/pi/conv_initial/bias/Initializer/zeros:08
�
'ppo_agent/ppo2_model/pi/conv_0/kernel:0,ppo_agent/ppo2_model/pi/conv_0/kernel/Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/read:02Bppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform:08
�
%ppo_agent/ppo2_model/pi/conv_0/bias:0*ppo_agent/ppo2_model/pi/conv_0/bias/Assign*ppo_agent/ppo2_model/pi/conv_0/bias/read:027ppo_agent/ppo2_model/pi/conv_0/bias/Initializer/zeros:08
�
'ppo_agent/ppo2_model/pi/conv_1/kernel:0,ppo_agent/ppo2_model/pi/conv_1/kernel/Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/read:02Bppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform:08
�
%ppo_agent/ppo2_model/pi/conv_1/bias:0*ppo_agent/ppo2_model/pi/conv_1/bias/Assign*ppo_agent/ppo2_model/pi/conv_1/bias/read:027ppo_agent/ppo2_model/pi/conv_1/bias/Initializer/zeros:08
�
&ppo_agent/ppo2_model/pi/dense/kernel:0+ppo_agent/ppo2_model/pi/dense/kernel/Assign+ppo_agent/ppo2_model/pi/dense/kernel/read:02Appo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform:08
�
$ppo_agent/ppo2_model/pi/dense/bias:0)ppo_agent/ppo2_model/pi/dense/bias/Assign)ppo_agent/ppo2_model/pi/dense/bias/read:026ppo_agent/ppo2_model/pi/dense/bias/Initializer/zeros:08
�
(ppo_agent/ppo2_model/pi/dense_1/kernel:0-ppo_agent/ppo2_model/pi/dense_1/kernel/Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/read:02Cppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform:08
�
&ppo_agent/ppo2_model/pi/dense_1/bias:0+ppo_agent/ppo2_model/pi/dense_1/bias/Assign+ppo_agent/ppo2_model/pi/dense_1/bias/read:028ppo_agent/ppo2_model/pi/dense_1/bias/Initializer/zeros:08
�
(ppo_agent/ppo2_model/pi/dense_2/kernel:0-ppo_agent/ppo2_model/pi/dense_2/kernel/Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/read:02Cppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform:08
�
&ppo_agent/ppo2_model/pi/dense_2/bias:0+ppo_agent/ppo2_model/pi/dense_2/bias/Assign+ppo_agent/ppo2_model/pi/dense_2/bias/read:028ppo_agent/ppo2_model/pi/dense_2/bias/Initializer/zeros:08
�
ppo_agent/ppo2_model/pi/w:0 ppo_agent/ppo2_model/pi/w/Assign ppo_agent/ppo2_model/pi/w/read:025ppo_agent/ppo2_model/pi/w/Initializer/initial_value:08
�
ppo_agent/ppo2_model/pi/b:0 ppo_agent/ppo2_model/pi/b/Assign ppo_agent/ppo2_model/pi/b/read:02-ppo_agent/ppo2_model/pi/b/Initializer/Const:08
�
ppo_agent/ppo2_model/vf/w:0 ppo_agent/ppo2_model/vf/w/Assign ppo_agent/ppo2_model/vf/w/read:025ppo_agent/ppo2_model/vf/w/Initializer/initial_value:08
�
ppo_agent/ppo2_model/vf/b:0 ppo_agent/ppo2_model/vf/b/Assign ppo_agent/ppo2_model/vf/b/read:02-ppo_agent/ppo2_model/vf/b/Initializer/Const:08"
train_op

Adam"�M
	variables�L�L
�
-ppo_agent/ppo2_model/pi/conv_initial/kernel:02ppo_agent/ppo2_model/pi/conv_initial/kernel/Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/read:02Hppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform:08
�
+ppo_agent/ppo2_model/pi/conv_initial/bias:00ppo_agent/ppo2_model/pi/conv_initial/bias/Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/read:02=ppo_agent/ppo2_model/pi/conv_initial/bias/Initializer/zeros:08
�
'ppo_agent/ppo2_model/pi/conv_0/kernel:0,ppo_agent/ppo2_model/pi/conv_0/kernel/Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/read:02Bppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform:08
�
%ppo_agent/ppo2_model/pi/conv_0/bias:0*ppo_agent/ppo2_model/pi/conv_0/bias/Assign*ppo_agent/ppo2_model/pi/conv_0/bias/read:027ppo_agent/ppo2_model/pi/conv_0/bias/Initializer/zeros:08
�
'ppo_agent/ppo2_model/pi/conv_1/kernel:0,ppo_agent/ppo2_model/pi/conv_1/kernel/Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/read:02Bppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform:08
�
%ppo_agent/ppo2_model/pi/conv_1/bias:0*ppo_agent/ppo2_model/pi/conv_1/bias/Assign*ppo_agent/ppo2_model/pi/conv_1/bias/read:027ppo_agent/ppo2_model/pi/conv_1/bias/Initializer/zeros:08
�
&ppo_agent/ppo2_model/pi/dense/kernel:0+ppo_agent/ppo2_model/pi/dense/kernel/Assign+ppo_agent/ppo2_model/pi/dense/kernel/read:02Appo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform:08
�
$ppo_agent/ppo2_model/pi/dense/bias:0)ppo_agent/ppo2_model/pi/dense/bias/Assign)ppo_agent/ppo2_model/pi/dense/bias/read:026ppo_agent/ppo2_model/pi/dense/bias/Initializer/zeros:08
�
(ppo_agent/ppo2_model/pi/dense_1/kernel:0-ppo_agent/ppo2_model/pi/dense_1/kernel/Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/read:02Cppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform:08
�
&ppo_agent/ppo2_model/pi/dense_1/bias:0+ppo_agent/ppo2_model/pi/dense_1/bias/Assign+ppo_agent/ppo2_model/pi/dense_1/bias/read:028ppo_agent/ppo2_model/pi/dense_1/bias/Initializer/zeros:08
�
(ppo_agent/ppo2_model/pi/dense_2/kernel:0-ppo_agent/ppo2_model/pi/dense_2/kernel/Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/read:02Cppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform:08
�
&ppo_agent/ppo2_model/pi/dense_2/bias:0+ppo_agent/ppo2_model/pi/dense_2/bias/Assign+ppo_agent/ppo2_model/pi/dense_2/bias/read:028ppo_agent/ppo2_model/pi/dense_2/bias/Initializer/zeros:08
�
ppo_agent/ppo2_model/pi/w:0 ppo_agent/ppo2_model/pi/w/Assign ppo_agent/ppo2_model/pi/w/read:025ppo_agent/ppo2_model/pi/w/Initializer/initial_value:08
�
ppo_agent/ppo2_model/pi/b:0 ppo_agent/ppo2_model/pi/b/Assign ppo_agent/ppo2_model/pi/b/read:02-ppo_agent/ppo2_model/pi/b/Initializer/Const:08
�
ppo_agent/ppo2_model/vf/w:0 ppo_agent/ppo2_model/vf/w/Assign ppo_agent/ppo2_model/vf/w/read:025ppo_agent/ppo2_model/vf/w/Initializer/initial_value:08
�
ppo_agent/ppo2_model/vf/b:0 ppo_agent/ppo2_model/vf/b/Assign ppo_agent/ppo2_model/vf/b/read:02-ppo_agent/ppo2_model/vf/b/Initializer/Const:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
�
2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam:07ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Assign7ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/read:02Dppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zeros:0
�
4ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1:09ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Assign9ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/read:02Fppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zeros:0
�
0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam:05ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/Assign5ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/read:02Bppo_agent/ppo2_model/pi/conv_initial/bias/Adam/Initializer/zeros:0
�
2ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1:07ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/Assign7ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/read:02Dppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/Initializer/zeros:0
�
,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam:01ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Assign1ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/read:02>ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zeros:0
�
.ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1:03ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Assign3ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/read:02@ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zeros:0
�
*ppo_agent/ppo2_model/pi/conv_0/bias/Adam:0/ppo_agent/ppo2_model/pi/conv_0/bias/Adam/Assign/ppo_agent/ppo2_model/pi/conv_0/bias/Adam/read:02<ppo_agent/ppo2_model/pi/conv_0/bias/Adam/Initializer/zeros:0
�
,ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1:01ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/Assign1ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/read:02>ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/Initializer/zeros:0
�
,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam:01ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Assign1ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/read:02>ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zeros:0
�
.ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1:03ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Assign3ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/read:02@ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zeros:0
�
*ppo_agent/ppo2_model/pi/conv_1/bias/Adam:0/ppo_agent/ppo2_model/pi/conv_1/bias/Adam/Assign/ppo_agent/ppo2_model/pi/conv_1/bias/Adam/read:02<ppo_agent/ppo2_model/pi/conv_1/bias/Adam/Initializer/zeros:0
�
,ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1:01ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/Assign1ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/read:02>ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/Initializer/zeros:0
�
+ppo_agent/ppo2_model/pi/dense/kernel/Adam:00ppo_agent/ppo2_model/pi/dense/kernel/Adam/Assign0ppo_agent/ppo2_model/pi/dense/kernel/Adam/read:02=ppo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zeros:0
�
-ppo_agent/ppo2_model/pi/dense/kernel/Adam_1:02ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Assign2ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/read:02?ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zeros:0
�
)ppo_agent/ppo2_model/pi/dense/bias/Adam:0.ppo_agent/ppo2_model/pi/dense/bias/Adam/Assign.ppo_agent/ppo2_model/pi/dense/bias/Adam/read:02;ppo_agent/ppo2_model/pi/dense/bias/Adam/Initializer/zeros:0
�
+ppo_agent/ppo2_model/pi/dense/bias/Adam_1:00ppo_agent/ppo2_model/pi/dense/bias/Adam_1/Assign0ppo_agent/ppo2_model/pi/dense/bias/Adam_1/read:02=ppo_agent/ppo2_model/pi/dense/bias/Adam_1/Initializer/zeros:0
�
-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam:02ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Assign2ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/read:02?ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zeros:0
�
/ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1:04ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Assign4ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/read:02Appo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zeros:0
�
+ppo_agent/ppo2_model/pi/dense_1/bias/Adam:00ppo_agent/ppo2_model/pi/dense_1/bias/Adam/Assign0ppo_agent/ppo2_model/pi/dense_1/bias/Adam/read:02=ppo_agent/ppo2_model/pi/dense_1/bias/Adam/Initializer/zeros:0
�
-ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1:02ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/Assign2ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/read:02?ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/Initializer/zeros:0
�
-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam:02ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Assign2ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/read:02?ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zeros:0
�
/ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1:04ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Assign4ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/read:02Appo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zeros:0
�
+ppo_agent/ppo2_model/pi/dense_2/bias/Adam:00ppo_agent/ppo2_model/pi/dense_2/bias/Adam/Assign0ppo_agent/ppo2_model/pi/dense_2/bias/Adam/read:02=ppo_agent/ppo2_model/pi/dense_2/bias/Adam/Initializer/zeros:0
�
-ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1:02ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/Assign2ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/read:02?ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/Initializer/zeros:0
�
 ppo_agent/ppo2_model/pi/w/Adam:0%ppo_agent/ppo2_model/pi/w/Adam/Assign%ppo_agent/ppo2_model/pi/w/Adam/read:022ppo_agent/ppo2_model/pi/w/Adam/Initializer/zeros:0
�
"ppo_agent/ppo2_model/pi/w/Adam_1:0'ppo_agent/ppo2_model/pi/w/Adam_1/Assign'ppo_agent/ppo2_model/pi/w/Adam_1/read:024ppo_agent/ppo2_model/pi/w/Adam_1/Initializer/zeros:0
�
 ppo_agent/ppo2_model/pi/b/Adam:0%ppo_agent/ppo2_model/pi/b/Adam/Assign%ppo_agent/ppo2_model/pi/b/Adam/read:022ppo_agent/ppo2_model/pi/b/Adam/Initializer/zeros:0
�
"ppo_agent/ppo2_model/pi/b/Adam_1:0'ppo_agent/ppo2_model/pi/b/Adam_1/Assign'ppo_agent/ppo2_model/pi/b/Adam_1/read:024ppo_agent/ppo2_model/pi/b/Adam_1/Initializer/zeros:0
�
 ppo_agent/ppo2_model/vf/w/Adam:0%ppo_agent/ppo2_model/vf/w/Adam/Assign%ppo_agent/ppo2_model/vf/w/Adam/read:022ppo_agent/ppo2_model/vf/w/Adam/Initializer/zeros:0
�
"ppo_agent/ppo2_model/vf/w/Adam_1:0'ppo_agent/ppo2_model/vf/w/Adam_1/Assign'ppo_agent/ppo2_model/vf/w/Adam_1/read:024ppo_agent/ppo2_model/vf/w/Adam_1/Initializer/zeros:0
�
 ppo_agent/ppo2_model/vf/b/Adam:0%ppo_agent/ppo2_model/vf/b/Adam/Assign%ppo_agent/ppo2_model/vf/b/Adam/read:022ppo_agent/ppo2_model/vf/b/Adam/Initializer/zeros:0
�
"ppo_agent/ppo2_model/vf/b/Adam_1:0'ppo_agent/ppo2_model/vf/b/Adam_1/Assign'ppo_agent/ppo2_model/vf/b/Adam_1/read:024ppo_agent/ppo2_model/vf/b/Adam_1/Initializer/zeros:0*�
serving_default�
6
obs/
ppo_agent/ppo2_model/Ob:0	/
value&
ppo_agent/ppo2_model/value:01
action'
ppo_agent/ppo2_model/action:0	A
action_probs1
#ppo_agent/ppo2_model/action_probs:0tensorflow/serving/predict