??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
3
Square
x"T
y"T"
Ttype:
2
	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
|
dense_572/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:[
*!
shared_namedense_572/kernel
u
$dense_572/kernel/Read/ReadVariableOpReadVariableOpdense_572/kernel*
_output_shapes

:[
*
dtype0
t
dense_572/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_572/bias
m
"dense_572/bias/Read/ReadVariableOpReadVariableOpdense_572/bias*
_output_shapes
:
*
dtype0
|
dense_573/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*!
shared_namedense_573/kernel
u
$dense_573/kernel/Read/ReadVariableOpReadVariableOpdense_573/kernel*
_output_shapes

:

*
dtype0
t
dense_573/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_573/bias
m
"dense_573/bias/Read/ReadVariableOpReadVariableOpdense_573/bias*
_output_shapes
:
*
dtype0
|
dense_574/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*!
shared_namedense_574/kernel
u
$dense_574/kernel/Read/ReadVariableOpReadVariableOpdense_574/kernel*
_output_shapes

:

*
dtype0
t
dense_574/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_574/bias
m
"dense_574/bias/Read/ReadVariableOpReadVariableOpdense_574/bias*
_output_shapes
:
*
dtype0
|
dense_575/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*!
shared_namedense_575/kernel
u
$dense_575/kernel/Read/ReadVariableOpReadVariableOpdense_575/kernel*
_output_shapes

:

*
dtype0
t
dense_575/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_575/bias
m
"dense_575/bias/Read/ReadVariableOpReadVariableOpdense_575/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_572/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:[
*(
shared_nameAdam/dense_572/kernel/m
?
+Adam/dense_572/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_572/kernel/m*
_output_shapes

:[
*
dtype0
?
Adam/dense_572/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_572/bias/m
{
)Adam/dense_572/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_572/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_573/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*(
shared_nameAdam/dense_573/kernel/m
?
+Adam/dense_573/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_573/kernel/m*
_output_shapes

:

*
dtype0
?
Adam/dense_573/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_573/bias/m
{
)Adam/dense_573/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_573/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_574/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*(
shared_nameAdam/dense_574/kernel/m
?
+Adam/dense_574/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_574/kernel/m*
_output_shapes

:

*
dtype0
?
Adam/dense_574/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_574/bias/m
{
)Adam/dense_574/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_574/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_575/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*(
shared_nameAdam/dense_575/kernel/m
?
+Adam/dense_575/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_575/kernel/m*
_output_shapes

:

*
dtype0
?
Adam/dense_575/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_575/bias/m
{
)Adam/dense_575/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_575/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_572/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:[
*(
shared_nameAdam/dense_572/kernel/v
?
+Adam/dense_572/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_572/kernel/v*
_output_shapes

:[
*
dtype0
?
Adam/dense_572/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_572/bias/v
{
)Adam/dense_572/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_572/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_573/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*(
shared_nameAdam/dense_573/kernel/v
?
+Adam/dense_573/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_573/kernel/v*
_output_shapes

:

*
dtype0
?
Adam/dense_573/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_573/bias/v
{
)Adam/dense_573/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_573/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_574/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*(
shared_nameAdam/dense_574/kernel/v
?
+Adam/dense_574/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_574/kernel/v*
_output_shapes

:

*
dtype0
?
Adam/dense_574/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_574/bias/v
{
)Adam/dense_574/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_574/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_575/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*(
shared_nameAdam/dense_575/kernel/v
?
+Adam/dense_575/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_575/kernel/v*
_output_shapes

:

*
dtype0
?
Adam/dense_575/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_575/bias/v
{
)Adam/dense_575/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_575/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?/
value?/B?/ B?/
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
?
(iter

)beta_1

*beta_2
	+decay
,learning_ratemVmWmXmYmZm[m\m]v^v_v`vavbvcvdve
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
	regularization_losses
 
\Z
VARIABLE_VALUEdense_572/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_572/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEdense_573/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_573/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEdense_574/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_574/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEdense_575/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_575/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
 	variables
!trainable_variables
"regularization_losses
 
 
 
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
$	variables
%trainable_variables
&regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4

K0
L1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Mtotal
	Ncount
O	variables
P	keras_api
D
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
U	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

O	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

T	variables
}
VARIABLE_VALUEAdam/dense_572/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_572/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_573/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_573/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_574/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_574/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_575/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_575/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_572/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_572/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_573/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_573/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_574/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_574/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_575/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_575/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_dense_572_inputPlaceholder*'
_output_shapes
:?????????[*
dtype0*
shape:?????????[
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_572_inputdense_572/kerneldense_572/biasdense_573/kerneldense_573/biasdense_574/kerneldense_574/biasdense_575/kerneldense_575/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_44352809
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_572/kernel/Read/ReadVariableOp"dense_572/bias/Read/ReadVariableOp$dense_573/kernel/Read/ReadVariableOp"dense_573/bias/Read/ReadVariableOp$dense_574/kernel/Read/ReadVariableOp"dense_574/bias/Read/ReadVariableOp$dense_575/kernel/Read/ReadVariableOp"dense_575/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_572/kernel/m/Read/ReadVariableOp)Adam/dense_572/bias/m/Read/ReadVariableOp+Adam/dense_573/kernel/m/Read/ReadVariableOp)Adam/dense_573/bias/m/Read/ReadVariableOp+Adam/dense_574/kernel/m/Read/ReadVariableOp)Adam/dense_574/bias/m/Read/ReadVariableOp+Adam/dense_575/kernel/m/Read/ReadVariableOp)Adam/dense_575/bias/m/Read/ReadVariableOp+Adam/dense_572/kernel/v/Read/ReadVariableOp)Adam/dense_572/bias/v/Read/ReadVariableOp+Adam/dense_573/kernel/v/Read/ReadVariableOp)Adam/dense_573/bias/v/Read/ReadVariableOp+Adam/dense_574/kernel/v/Read/ReadVariableOp)Adam/dense_574/bias/v/Read/ReadVariableOp+Adam/dense_575/kernel/v/Read/ReadVariableOp)Adam/dense_575/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_44353455
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_572/kerneldense_572/biasdense_573/kerneldense_573/biasdense_574/kerneldense_574/biasdense_575/kerneldense_575/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_572/kernel/mAdam/dense_572/bias/mAdam/dense_573/kernel/mAdam/dense_573/bias/mAdam/dense_574/kernel/mAdam/dense_574/bias/mAdam/dense_575/kernel/mAdam/dense_575/bias/mAdam/dense_572/kernel/vAdam/dense_572/bias/vAdam/dense_573/kernel/vAdam/dense_573/bias/vAdam/dense_574/kernel/vAdam/dense_574/bias/vAdam/dense_575/kernel/vAdam/dense_575/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_44353564ɏ
?
J
3__inference_dense_574_activity_regularizer_44352119
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????G
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
?
?
__inference_loss_fn_1_44353220G
9dense_572_bias_regularizer_square_readvariableop_resource:

identity??0dense_572/bias/Regularizer/Square/ReadVariableOp?
0dense_572/bias/Regularizer/Square/ReadVariableOpReadVariableOp9dense_572_bias_regularizer_square_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_572/bias/Regularizer/SquareSquare8dense_572/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_572/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_572/bias/Regularizer/SumSum%dense_572/bias/Regularizer/Square:y:0)dense_572/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_572/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_572/bias/Regularizer/mulMul)dense_572/bias/Regularizer/mul/x:output:0'dense_572/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_572/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^dense_572/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_572/bias/Regularizer/Square/ReadVariableOp0dense_572/bias/Regularizer/Square/ReadVariableOp
?
?
,__inference_dense_573_layer_call_fn_44353126

inputs
unknown:


	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_573_layer_call_and_return_conditional_losses_44352187o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_44353209M
;dense_572_kernel_regularizer_square_readvariableop_resource:[

identity??2dense_572/kernel/Regularizer/Square/ReadVariableOp?
2dense_572/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_572_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:[
*
dtype0?
#dense_572/kernel/Regularizer/SquareSquare:dense_572/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:[
s
"dense_572/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_572/kernel/Regularizer/SumSum'dense_572/kernel/Regularizer/Square:y:0+dense_572/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_572/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_572/kernel/Regularizer/mulMul+dense_572/kernel/Regularizer/mul/x:output:0)dense_572/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_572/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_572/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_572/kernel/Regularizer/Square/ReadVariableOp2dense_572/kernel/Regularizer/Square/ReadVariableOp
?
?
G__inference_dense_573_layer_call_and_return_conditional_losses_44353310

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_573/bias/Regularizer/Square/ReadVariableOp?2dense_573/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
2dense_573/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
#dense_573/kernel/Regularizer/SquareSquare:dense_573/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_573/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_573/kernel/Regularizer/SumSum'dense_573/kernel/Regularizer/Square:y:0+dense_573/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_573/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_573/kernel/Regularizer/mulMul+dense_573/kernel/Regularizer/mul/x:output:0)dense_573/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0dense_573/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_573/bias/Regularizer/SquareSquare8dense_573/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_573/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_573/bias/Regularizer/SumSum%dense_573/bias/Regularizer/Square:y:0)dense_573/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_573/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_573/bias/Regularizer/mulMul)dense_573/bias/Regularizer/mul/x:output:0'dense_573/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_573/bias/Regularizer/Square/ReadVariableOp3^dense_573/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_573/bias/Regularizer/Square/ReadVariableOp0dense_573/bias/Regularizer/Square/ReadVariableOp2h
2dense_573/kernel/Regularizer/Square/ReadVariableOp2dense_573/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
G__inference_dense_572_layer_call_and_return_conditional_losses_44353287

inputs0
matmul_readvariableop_resource:[
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_572/bias/Regularizer/Square/ReadVariableOp?2dense_572/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:[
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
2dense_572/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:[
*
dtype0?
#dense_572/kernel/Regularizer/SquareSquare:dense_572/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:[
s
"dense_572/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_572/kernel/Regularizer/SumSum'dense_572/kernel/Regularizer/Square:y:0+dense_572/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_572/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_572/kernel/Regularizer/mulMul+dense_572/kernel/Regularizer/mul/x:output:0)dense_572/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0dense_572/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_572/bias/Regularizer/SquareSquare8dense_572/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_572/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_572/bias/Regularizer/SumSum%dense_572/bias/Regularizer/Square:y:0)dense_572/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_572/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_572/bias/Regularizer/mulMul)dense_572/bias/Regularizer/mul/x:output:0'dense_572/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_572/bias/Regularizer/Square/ReadVariableOp3^dense_572/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????[: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_572/bias/Regularizer/Square/ReadVariableOp0dense_572/bias/Regularizer/Square/ReadVariableOp2h
2dense_572/kernel/Regularizer/Square/ReadVariableOp2dense_572/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????[
 
_user_specified_nameinputs
?
?
G__inference_dense_574_layer_call_and_return_conditional_losses_44353333

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_574/bias/Regularizer/Square/ReadVariableOp?2dense_574/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
2dense_574/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
#dense_574/kernel/Regularizer/SquareSquare:dense_574/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_574/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_574/kernel/Regularizer/SumSum'dense_574/kernel/Regularizer/Square:y:0+dense_574/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_574/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_574/kernel/Regularizer/mulMul+dense_574/kernel/Regularizer/mul/x:output:0)dense_574/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0dense_574/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_574/bias/Regularizer/SquareSquare8dense_574/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_574/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_574/bias/Regularizer/SumSum%dense_574/bias/Regularizer/Square:y:0)dense_574/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_574/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_574/bias/Regularizer/mulMul)dense_574/bias/Regularizer/mul/x:output:0'dense_574/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_574/bias/Regularizer/Square/ReadVariableOp3^dense_574/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_574/bias/Regularizer/Square/ReadVariableOp0dense_574/bias/Regularizer/Square/ReadVariableOp2h
2dense_574/kernel/Regularizer/Square/ReadVariableOp2dense_574/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
1__inference_sequential_143_layer_call_fn_44352857

inputs
unknown:[

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:


	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????
: : : **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352520o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????[: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????[
 
_user_specified_nameinputs
?r
?
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352520

inputs$
dense_572_44352435:[
 
dense_572_44352437:
$
dense_573_44352448:

 
dense_573_44352450:
$
dense_574_44352461:

 
dense_574_44352463:
$
dense_575_44352474:

 
dense_575_44352476:

identity

identity_1

identity_2

identity_3??!dense_572/StatefulPartitionedCall?0dense_572/bias/Regularizer/Square/ReadVariableOp?2dense_572/kernel/Regularizer/Square/ReadVariableOp?!dense_573/StatefulPartitionedCall?0dense_573/bias/Regularizer/Square/ReadVariableOp?2dense_573/kernel/Regularizer/Square/ReadVariableOp?!dense_574/StatefulPartitionedCall?0dense_574/bias/Regularizer/Square/ReadVariableOp?2dense_574/kernel/Regularizer/Square/ReadVariableOp?!dense_575/StatefulPartitionedCall_
dense_572/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????[?
!dense_572/StatefulPartitionedCallStatefulPartitionedCalldense_572/Cast:y:0dense_572_44352435dense_572_44352437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_572_layer_call_and_return_conditional_losses_44352150?
-dense_572/ActivityRegularizer/PartitionedCallPartitionedCall*dense_572/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_572_activity_regularizer_44352093}
#dense_572/ActivityRegularizer/ShapeShape*dense_572/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1dense_572/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_572/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_572/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_572/ActivityRegularizer/strided_sliceStridedSlice,dense_572/ActivityRegularizer/Shape:output:0:dense_572/ActivityRegularizer/strided_slice/stack:output:0<dense_572/ActivityRegularizer/strided_slice/stack_1:output:0<dense_572/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_572/ActivityRegularizer/CastCast4dense_572/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_572/ActivityRegularizer/truedivRealDiv6dense_572/ActivityRegularizer/PartitionedCall:output:0&dense_572/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!dense_573/StatefulPartitionedCallStatefulPartitionedCall*dense_572/StatefulPartitionedCall:output:0dense_573_44352448dense_573_44352450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_573_layer_call_and_return_conditional_losses_44352187?
-dense_573/ActivityRegularizer/PartitionedCallPartitionedCall*dense_573/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_573_activity_regularizer_44352106}
#dense_573/ActivityRegularizer/ShapeShape*dense_573/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1dense_573/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_573/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_573/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_573/ActivityRegularizer/strided_sliceStridedSlice,dense_573/ActivityRegularizer/Shape:output:0:dense_573/ActivityRegularizer/strided_slice/stack:output:0<dense_573/ActivityRegularizer/strided_slice/stack_1:output:0<dense_573/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_573/ActivityRegularizer/CastCast4dense_573/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_573/ActivityRegularizer/truedivRealDiv6dense_573/ActivityRegularizer/PartitionedCall:output:0&dense_573/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!dense_574/StatefulPartitionedCallStatefulPartitionedCall*dense_573/StatefulPartitionedCall:output:0dense_574_44352461dense_574_44352463*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_574_layer_call_and_return_conditional_losses_44352224?
-dense_574/ActivityRegularizer/PartitionedCallPartitionedCall*dense_574/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_574_activity_regularizer_44352119}
#dense_574/ActivityRegularizer/ShapeShape*dense_574/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1dense_574/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_574/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_574/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_574/ActivityRegularizer/strided_sliceStridedSlice,dense_574/ActivityRegularizer/Shape:output:0:dense_574/ActivityRegularizer/strided_slice/stack:output:0<dense_574/ActivityRegularizer/strided_slice/stack_1:output:0<dense_574/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_574/ActivityRegularizer/CastCast4dense_574/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_574/ActivityRegularizer/truedivRealDiv6dense_574/ActivityRegularizer/PartitionedCall:output:0&dense_574/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!dense_575/StatefulPartitionedCallStatefulPartitionedCall*dense_574/StatefulPartitionedCall:output:0dense_575_44352474dense_575_44352476*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_575_layer_call_and_return_conditional_losses_44352248?
softmax_143/PartitionedCallPartitionedCall*dense_575/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_softmax_143_layer_call_and_return_conditional_losses_44352259?
2dense_572/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_572_44352435*
_output_shapes

:[
*
dtype0?
#dense_572/kernel/Regularizer/SquareSquare:dense_572/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:[
s
"dense_572/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_572/kernel/Regularizer/SumSum'dense_572/kernel/Regularizer/Square:y:0+dense_572/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_572/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_572/kernel/Regularizer/mulMul+dense_572/kernel/Regularizer/mul/x:output:0)dense_572/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_572/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_572_44352437*
_output_shapes
:
*
dtype0?
!dense_572/bias/Regularizer/SquareSquare8dense_572/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_572/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_572/bias/Regularizer/SumSum%dense_572/bias/Regularizer/Square:y:0)dense_572/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_572/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_572/bias/Regularizer/mulMul)dense_572/bias/Regularizer/mul/x:output:0'dense_572/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_573/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_573_44352448*
_output_shapes

:

*
dtype0?
#dense_573/kernel/Regularizer/SquareSquare:dense_573/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_573/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_573/kernel/Regularizer/SumSum'dense_573/kernel/Regularizer/Square:y:0+dense_573/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_573/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_573/kernel/Regularizer/mulMul+dense_573/kernel/Regularizer/mul/x:output:0)dense_573/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_573/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_573_44352450*
_output_shapes
:
*
dtype0?
!dense_573/bias/Regularizer/SquareSquare8dense_573/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_573/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_573/bias/Regularizer/SumSum%dense_573/bias/Regularizer/Square:y:0)dense_573/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_573/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_573/bias/Regularizer/mulMul)dense_573/bias/Regularizer/mul/x:output:0'dense_573/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_574/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_574_44352461*
_output_shapes

:

*
dtype0?
#dense_574/kernel/Regularizer/SquareSquare:dense_574/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_574/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_574/kernel/Regularizer/SumSum'dense_574/kernel/Regularizer/Square:y:0+dense_574/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_574/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_574/kernel/Regularizer/mulMul+dense_574/kernel/Regularizer/mul/x:output:0)dense_574/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_574/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_574_44352463*
_output_shapes
:
*
dtype0?
!dense_574/bias/Regularizer/SquareSquare8dense_574/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_574/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_574/bias/Regularizer/SumSum%dense_574/bias/Regularizer/Square:y:0)dense_574/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_574/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_574/bias/Regularizer/mulMul)dense_574/bias/Regularizer/mul/x:output:0'dense_574/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
IdentityIdentity$softmax_143/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
i

Identity_1Identity)dense_572/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_2Identity)dense_573/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_3Identity)dense_574/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp"^dense_572/StatefulPartitionedCall1^dense_572/bias/Regularizer/Square/ReadVariableOp3^dense_572/kernel/Regularizer/Square/ReadVariableOp"^dense_573/StatefulPartitionedCall1^dense_573/bias/Regularizer/Square/ReadVariableOp3^dense_573/kernel/Regularizer/Square/ReadVariableOp"^dense_574/StatefulPartitionedCall1^dense_574/bias/Regularizer/Square/ReadVariableOp3^dense_574/kernel/Regularizer/Square/ReadVariableOp"^dense_575/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????[: : : : : : : : 2F
!dense_572/StatefulPartitionedCall!dense_572/StatefulPartitionedCall2d
0dense_572/bias/Regularizer/Square/ReadVariableOp0dense_572/bias/Regularizer/Square/ReadVariableOp2h
2dense_572/kernel/Regularizer/Square/ReadVariableOp2dense_572/kernel/Regularizer/Square/ReadVariableOp2F
!dense_573/StatefulPartitionedCall!dense_573/StatefulPartitionedCall2d
0dense_573/bias/Regularizer/Square/ReadVariableOp0dense_573/bias/Regularizer/Square/ReadVariableOp2h
2dense_573/kernel/Regularizer/Square/ReadVariableOp2dense_573/kernel/Regularizer/Square/ReadVariableOp2F
!dense_574/StatefulPartitionedCall!dense_574/StatefulPartitionedCall2d
0dense_574/bias/Regularizer/Square/ReadVariableOp0dense_574/bias/Regularizer/Square/ReadVariableOp2h
2dense_574/kernel/Regularizer/Square/ReadVariableOp2dense_574/kernel/Regularizer/Square/ReadVariableOp2F
!dense_575/StatefulPartitionedCall!dense_575/StatefulPartitionedCall:O K
'
_output_shapes
:?????????[
 
_user_specified_nameinputs
?
J
3__inference_dense_573_activity_regularizer_44352106
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????G
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
??
?	
L__inference_sequential_143_layer_call_and_return_conditional_losses_44353073

inputs:
(dense_572_matmul_readvariableop_resource:[
7
)dense_572_biasadd_readvariableop_resource:
:
(dense_573_matmul_readvariableop_resource:

7
)dense_573_biasadd_readvariableop_resource:
:
(dense_574_matmul_readvariableop_resource:

7
)dense_574_biasadd_readvariableop_resource:
:
(dense_575_matmul_readvariableop_resource:

7
)dense_575_biasadd_readvariableop_resource:

identity

identity_1

identity_2

identity_3?? dense_572/BiasAdd/ReadVariableOp?dense_572/MatMul/ReadVariableOp?0dense_572/bias/Regularizer/Square/ReadVariableOp?2dense_572/kernel/Regularizer/Square/ReadVariableOp? dense_573/BiasAdd/ReadVariableOp?dense_573/MatMul/ReadVariableOp?0dense_573/bias/Regularizer/Square/ReadVariableOp?2dense_573/kernel/Regularizer/Square/ReadVariableOp? dense_574/BiasAdd/ReadVariableOp?dense_574/MatMul/ReadVariableOp?0dense_574/bias/Regularizer/Square/ReadVariableOp?2dense_574/kernel/Regularizer/Square/ReadVariableOp? dense_575/BiasAdd/ReadVariableOp?dense_575/MatMul/ReadVariableOp_
dense_572/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????[?
dense_572/MatMul/ReadVariableOpReadVariableOp(dense_572_matmul_readvariableop_resource*
_output_shapes

:[
*
dtype0?
dense_572/MatMulMatMuldense_572/Cast:y:0'dense_572/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 dense_572/BiasAdd/ReadVariableOpReadVariableOp)dense_572_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_572/BiasAddBiasAdddense_572/MatMul:product:0(dense_572/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
d
dense_572/ReluReludense_572/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
~
$dense_572/ActivityRegularizer/SquareSquaredense_572/Relu:activations:0*
T0*'
_output_shapes
:?????????
t
#dense_572/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_572/ActivityRegularizer/SumSum(dense_572/ActivityRegularizer/Square:y:0,dense_572/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#dense_572/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!dense_572/ActivityRegularizer/mulMul,dense_572/ActivityRegularizer/mul/x:output:0*dense_572/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: o
#dense_572/ActivityRegularizer/ShapeShapedense_572/Relu:activations:0*
T0*
_output_shapes
:{
1dense_572/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_572/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_572/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_572/ActivityRegularizer/strided_sliceStridedSlice,dense_572/ActivityRegularizer/Shape:output:0:dense_572/ActivityRegularizer/strided_slice/stack:output:0<dense_572/ActivityRegularizer/strided_slice/stack_1:output:0<dense_572/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_572/ActivityRegularizer/CastCast4dense_572/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_572/ActivityRegularizer/truedivRealDiv%dense_572/ActivityRegularizer/mul:z:0&dense_572/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
dense_573/MatMul/ReadVariableOpReadVariableOp(dense_573_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
dense_573/MatMulMatMuldense_572/Relu:activations:0'dense_573/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 dense_573/BiasAdd/ReadVariableOpReadVariableOp)dense_573_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_573/BiasAddBiasAdddense_573/MatMul:product:0(dense_573/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
d
dense_573/ReluReludense_573/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
~
$dense_573/ActivityRegularizer/SquareSquaredense_573/Relu:activations:0*
T0*'
_output_shapes
:?????????
t
#dense_573/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_573/ActivityRegularizer/SumSum(dense_573/ActivityRegularizer/Square:y:0,dense_573/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#dense_573/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!dense_573/ActivityRegularizer/mulMul,dense_573/ActivityRegularizer/mul/x:output:0*dense_573/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: o
#dense_573/ActivityRegularizer/ShapeShapedense_573/Relu:activations:0*
T0*
_output_shapes
:{
1dense_573/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_573/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_573/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_573/ActivityRegularizer/strided_sliceStridedSlice,dense_573/ActivityRegularizer/Shape:output:0:dense_573/ActivityRegularizer/strided_slice/stack:output:0<dense_573/ActivityRegularizer/strided_slice/stack_1:output:0<dense_573/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_573/ActivityRegularizer/CastCast4dense_573/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_573/ActivityRegularizer/truedivRealDiv%dense_573/ActivityRegularizer/mul:z:0&dense_573/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
dense_574/MatMul/ReadVariableOpReadVariableOp(dense_574_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
dense_574/MatMulMatMuldense_573/Relu:activations:0'dense_574/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 dense_574/BiasAdd/ReadVariableOpReadVariableOp)dense_574_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_574/BiasAddBiasAdddense_574/MatMul:product:0(dense_574/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
d
dense_574/ReluReludense_574/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
~
$dense_574/ActivityRegularizer/SquareSquaredense_574/Relu:activations:0*
T0*'
_output_shapes
:?????????
t
#dense_574/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_574/ActivityRegularizer/SumSum(dense_574/ActivityRegularizer/Square:y:0,dense_574/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#dense_574/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!dense_574/ActivityRegularizer/mulMul,dense_574/ActivityRegularizer/mul/x:output:0*dense_574/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: o
#dense_574/ActivityRegularizer/ShapeShapedense_574/Relu:activations:0*
T0*
_output_shapes
:{
1dense_574/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_574/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_574/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_574/ActivityRegularizer/strided_sliceStridedSlice,dense_574/ActivityRegularizer/Shape:output:0:dense_574/ActivityRegularizer/strided_slice/stack:output:0<dense_574/ActivityRegularizer/strided_slice/stack_1:output:0<dense_574/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_574/ActivityRegularizer/CastCast4dense_574/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_574/ActivityRegularizer/truedivRealDiv%dense_574/ActivityRegularizer/mul:z:0&dense_574/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
dense_575/MatMul/ReadVariableOpReadVariableOp(dense_575_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
dense_575/MatMulMatMuldense_574/Relu:activations:0'dense_575/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 dense_575/BiasAdd/ReadVariableOpReadVariableOp)dense_575_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_575/BiasAddBiasAdddense_575/MatMul:product:0(dense_575/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
l
softmax_143/SoftmaxSoftmaxdense_575/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
2dense_572/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_572_matmul_readvariableop_resource*
_output_shapes

:[
*
dtype0?
#dense_572/kernel/Regularizer/SquareSquare:dense_572/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:[
s
"dense_572/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_572/kernel/Regularizer/SumSum'dense_572/kernel/Regularizer/Square:y:0+dense_572/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_572/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_572/kernel/Regularizer/mulMul+dense_572/kernel/Regularizer/mul/x:output:0)dense_572/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0dense_572/bias/Regularizer/Square/ReadVariableOpReadVariableOp)dense_572_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_572/bias/Regularizer/SquareSquare8dense_572/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_572/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_572/bias/Regularizer/SumSum%dense_572/bias/Regularizer/Square:y:0)dense_572/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_572/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_572/bias/Regularizer/mulMul)dense_572/bias/Regularizer/mul/x:output:0'dense_572/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_573/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_573_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
#dense_573/kernel/Regularizer/SquareSquare:dense_573/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_573/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_573/kernel/Regularizer/SumSum'dense_573/kernel/Regularizer/Square:y:0+dense_573/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_573/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_573/kernel/Regularizer/mulMul+dense_573/kernel/Regularizer/mul/x:output:0)dense_573/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0dense_573/bias/Regularizer/Square/ReadVariableOpReadVariableOp)dense_573_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_573/bias/Regularizer/SquareSquare8dense_573/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_573/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_573/bias/Regularizer/SumSum%dense_573/bias/Regularizer/Square:y:0)dense_573/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_573/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_573/bias/Regularizer/mulMul)dense_573/bias/Regularizer/mul/x:output:0'dense_573/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_574/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_574_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
#dense_574/kernel/Regularizer/SquareSquare:dense_574/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_574/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_574/kernel/Regularizer/SumSum'dense_574/kernel/Regularizer/Square:y:0+dense_574/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_574/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_574/kernel/Regularizer/mulMul+dense_574/kernel/Regularizer/mul/x:output:0)dense_574/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0dense_574/bias/Regularizer/Square/ReadVariableOpReadVariableOp)dense_574_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_574/bias/Regularizer/SquareSquare8dense_574/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_574/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_574/bias/Regularizer/SumSum%dense_574/bias/Regularizer/Square:y:0)dense_574/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_574/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_574/bias/Regularizer/mulMul)dense_574/bias/Regularizer/mul/x:output:0'dense_574/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: l
IdentityIdentitysoftmax_143/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
i

Identity_1Identity)dense_572/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_2Identity)dense_573/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_3Identity)dense_574/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp!^dense_572/BiasAdd/ReadVariableOp ^dense_572/MatMul/ReadVariableOp1^dense_572/bias/Regularizer/Square/ReadVariableOp3^dense_572/kernel/Regularizer/Square/ReadVariableOp!^dense_573/BiasAdd/ReadVariableOp ^dense_573/MatMul/ReadVariableOp1^dense_573/bias/Regularizer/Square/ReadVariableOp3^dense_573/kernel/Regularizer/Square/ReadVariableOp!^dense_574/BiasAdd/ReadVariableOp ^dense_574/MatMul/ReadVariableOp1^dense_574/bias/Regularizer/Square/ReadVariableOp3^dense_574/kernel/Regularizer/Square/ReadVariableOp!^dense_575/BiasAdd/ReadVariableOp ^dense_575/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????[: : : : : : : : 2D
 dense_572/BiasAdd/ReadVariableOp dense_572/BiasAdd/ReadVariableOp2B
dense_572/MatMul/ReadVariableOpdense_572/MatMul/ReadVariableOp2d
0dense_572/bias/Regularizer/Square/ReadVariableOp0dense_572/bias/Regularizer/Square/ReadVariableOp2h
2dense_572/kernel/Regularizer/Square/ReadVariableOp2dense_572/kernel/Regularizer/Square/ReadVariableOp2D
 dense_573/BiasAdd/ReadVariableOp dense_573/BiasAdd/ReadVariableOp2B
dense_573/MatMul/ReadVariableOpdense_573/MatMul/ReadVariableOp2d
0dense_573/bias/Regularizer/Square/ReadVariableOp0dense_573/bias/Regularizer/Square/ReadVariableOp2h
2dense_573/kernel/Regularizer/Square/ReadVariableOp2dense_573/kernel/Regularizer/Square/ReadVariableOp2D
 dense_574/BiasAdd/ReadVariableOp dense_574/BiasAdd/ReadVariableOp2B
dense_574/MatMul/ReadVariableOpdense_574/MatMul/ReadVariableOp2d
0dense_574/bias/Regularizer/Square/ReadVariableOp0dense_574/bias/Regularizer/Square/ReadVariableOp2h
2dense_574/kernel/Regularizer/Square/ReadVariableOp2dense_574/kernel/Regularizer/Square/ReadVariableOp2D
 dense_575/BiasAdd/ReadVariableOp dense_575/BiasAdd/ReadVariableOp2B
dense_575/MatMul/ReadVariableOpdense_575/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????[
 
_user_specified_nameinputs
?
e
I__inference_softmax_143_layer_call_and_return_conditional_losses_44353198

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?r
?
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352301

inputs$
dense_572_44352151:[
 
dense_572_44352153:
$
dense_573_44352188:

 
dense_573_44352190:
$
dense_574_44352225:

 
dense_574_44352227:
$
dense_575_44352249:

 
dense_575_44352251:

identity

identity_1

identity_2

identity_3??!dense_572/StatefulPartitionedCall?0dense_572/bias/Regularizer/Square/ReadVariableOp?2dense_572/kernel/Regularizer/Square/ReadVariableOp?!dense_573/StatefulPartitionedCall?0dense_573/bias/Regularizer/Square/ReadVariableOp?2dense_573/kernel/Regularizer/Square/ReadVariableOp?!dense_574/StatefulPartitionedCall?0dense_574/bias/Regularizer/Square/ReadVariableOp?2dense_574/kernel/Regularizer/Square/ReadVariableOp?!dense_575/StatefulPartitionedCall_
dense_572/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????[?
!dense_572/StatefulPartitionedCallStatefulPartitionedCalldense_572/Cast:y:0dense_572_44352151dense_572_44352153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_572_layer_call_and_return_conditional_losses_44352150?
-dense_572/ActivityRegularizer/PartitionedCallPartitionedCall*dense_572/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_572_activity_regularizer_44352093}
#dense_572/ActivityRegularizer/ShapeShape*dense_572/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1dense_572/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_572/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_572/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_572/ActivityRegularizer/strided_sliceStridedSlice,dense_572/ActivityRegularizer/Shape:output:0:dense_572/ActivityRegularizer/strided_slice/stack:output:0<dense_572/ActivityRegularizer/strided_slice/stack_1:output:0<dense_572/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_572/ActivityRegularizer/CastCast4dense_572/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_572/ActivityRegularizer/truedivRealDiv6dense_572/ActivityRegularizer/PartitionedCall:output:0&dense_572/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!dense_573/StatefulPartitionedCallStatefulPartitionedCall*dense_572/StatefulPartitionedCall:output:0dense_573_44352188dense_573_44352190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_573_layer_call_and_return_conditional_losses_44352187?
-dense_573/ActivityRegularizer/PartitionedCallPartitionedCall*dense_573/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_573_activity_regularizer_44352106}
#dense_573/ActivityRegularizer/ShapeShape*dense_573/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1dense_573/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_573/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_573/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_573/ActivityRegularizer/strided_sliceStridedSlice,dense_573/ActivityRegularizer/Shape:output:0:dense_573/ActivityRegularizer/strided_slice/stack:output:0<dense_573/ActivityRegularizer/strided_slice/stack_1:output:0<dense_573/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_573/ActivityRegularizer/CastCast4dense_573/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_573/ActivityRegularizer/truedivRealDiv6dense_573/ActivityRegularizer/PartitionedCall:output:0&dense_573/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!dense_574/StatefulPartitionedCallStatefulPartitionedCall*dense_573/StatefulPartitionedCall:output:0dense_574_44352225dense_574_44352227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_574_layer_call_and_return_conditional_losses_44352224?
-dense_574/ActivityRegularizer/PartitionedCallPartitionedCall*dense_574/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_574_activity_regularizer_44352119}
#dense_574/ActivityRegularizer/ShapeShape*dense_574/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1dense_574/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_574/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_574/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_574/ActivityRegularizer/strided_sliceStridedSlice,dense_574/ActivityRegularizer/Shape:output:0:dense_574/ActivityRegularizer/strided_slice/stack:output:0<dense_574/ActivityRegularizer/strided_slice/stack_1:output:0<dense_574/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_574/ActivityRegularizer/CastCast4dense_574/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_574/ActivityRegularizer/truedivRealDiv6dense_574/ActivityRegularizer/PartitionedCall:output:0&dense_574/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!dense_575/StatefulPartitionedCallStatefulPartitionedCall*dense_574/StatefulPartitionedCall:output:0dense_575_44352249dense_575_44352251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_575_layer_call_and_return_conditional_losses_44352248?
softmax_143/PartitionedCallPartitionedCall*dense_575/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_softmax_143_layer_call_and_return_conditional_losses_44352259?
2dense_572/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_572_44352151*
_output_shapes

:[
*
dtype0?
#dense_572/kernel/Regularizer/SquareSquare:dense_572/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:[
s
"dense_572/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_572/kernel/Regularizer/SumSum'dense_572/kernel/Regularizer/Square:y:0+dense_572/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_572/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_572/kernel/Regularizer/mulMul+dense_572/kernel/Regularizer/mul/x:output:0)dense_572/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_572/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_572_44352153*
_output_shapes
:
*
dtype0?
!dense_572/bias/Regularizer/SquareSquare8dense_572/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_572/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_572/bias/Regularizer/SumSum%dense_572/bias/Regularizer/Square:y:0)dense_572/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_572/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_572/bias/Regularizer/mulMul)dense_572/bias/Regularizer/mul/x:output:0'dense_572/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_573/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_573_44352188*
_output_shapes

:

*
dtype0?
#dense_573/kernel/Regularizer/SquareSquare:dense_573/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_573/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_573/kernel/Regularizer/SumSum'dense_573/kernel/Regularizer/Square:y:0+dense_573/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_573/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_573/kernel/Regularizer/mulMul+dense_573/kernel/Regularizer/mul/x:output:0)dense_573/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_573/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_573_44352190*
_output_shapes
:
*
dtype0?
!dense_573/bias/Regularizer/SquareSquare8dense_573/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_573/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_573/bias/Regularizer/SumSum%dense_573/bias/Regularizer/Square:y:0)dense_573/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_573/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_573/bias/Regularizer/mulMul)dense_573/bias/Regularizer/mul/x:output:0'dense_573/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_574/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_574_44352225*
_output_shapes

:

*
dtype0?
#dense_574/kernel/Regularizer/SquareSquare:dense_574/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_574/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_574/kernel/Regularizer/SumSum'dense_574/kernel/Regularizer/Square:y:0+dense_574/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_574/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_574/kernel/Regularizer/mulMul+dense_574/kernel/Regularizer/mul/x:output:0)dense_574/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_574/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_574_44352227*
_output_shapes
:
*
dtype0?
!dense_574/bias/Regularizer/SquareSquare8dense_574/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_574/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_574/bias/Regularizer/SumSum%dense_574/bias/Regularizer/Square:y:0)dense_574/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_574/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_574/bias/Regularizer/mulMul)dense_574/bias/Regularizer/mul/x:output:0'dense_574/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
IdentityIdentity$softmax_143/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
i

Identity_1Identity)dense_572/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_2Identity)dense_573/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_3Identity)dense_574/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp"^dense_572/StatefulPartitionedCall1^dense_572/bias/Regularizer/Square/ReadVariableOp3^dense_572/kernel/Regularizer/Square/ReadVariableOp"^dense_573/StatefulPartitionedCall1^dense_573/bias/Regularizer/Square/ReadVariableOp3^dense_573/kernel/Regularizer/Square/ReadVariableOp"^dense_574/StatefulPartitionedCall1^dense_574/bias/Regularizer/Square/ReadVariableOp3^dense_574/kernel/Regularizer/Square/ReadVariableOp"^dense_575/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????[: : : : : : : : 2F
!dense_572/StatefulPartitionedCall!dense_572/StatefulPartitionedCall2d
0dense_572/bias/Regularizer/Square/ReadVariableOp0dense_572/bias/Regularizer/Square/ReadVariableOp2h
2dense_572/kernel/Regularizer/Square/ReadVariableOp2dense_572/kernel/Regularizer/Square/ReadVariableOp2F
!dense_573/StatefulPartitionedCall!dense_573/StatefulPartitionedCall2d
0dense_573/bias/Regularizer/Square/ReadVariableOp0dense_573/bias/Regularizer/Square/ReadVariableOp2h
2dense_573/kernel/Regularizer/Square/ReadVariableOp2dense_573/kernel/Regularizer/Square/ReadVariableOp2F
!dense_574/StatefulPartitionedCall!dense_574/StatefulPartitionedCall2d
0dense_574/bias/Regularizer/Square/ReadVariableOp0dense_574/bias/Regularizer/Square/ReadVariableOp2h
2dense_574/kernel/Regularizer/Square/ReadVariableOp2dense_574/kernel/Regularizer/Square/ReadVariableOp2F
!dense_575/StatefulPartitionedCall!dense_575/StatefulPartitionedCall:O K
'
_output_shapes
:?????????[
 
_user_specified_nameinputs
?
?
K__inference_dense_573_layer_call_and_return_all_conditional_losses_44353137

inputs
unknown:


	unknown_0:

identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_573_layer_call_and_return_conditional_losses_44352187?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_573_activity_regularizer_44352106o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?r
?
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352744
dense_572_input$
dense_572_44352659:[
 
dense_572_44352661:
$
dense_573_44352672:

 
dense_573_44352674:
$
dense_574_44352685:

 
dense_574_44352687:
$
dense_575_44352698:

 
dense_575_44352700:

identity

identity_1

identity_2

identity_3??!dense_572/StatefulPartitionedCall?0dense_572/bias/Regularizer/Square/ReadVariableOp?2dense_572/kernel/Regularizer/Square/ReadVariableOp?!dense_573/StatefulPartitionedCall?0dense_573/bias/Regularizer/Square/ReadVariableOp?2dense_573/kernel/Regularizer/Square/ReadVariableOp?!dense_574/StatefulPartitionedCall?0dense_574/bias/Regularizer/Square/ReadVariableOp?2dense_574/kernel/Regularizer/Square/ReadVariableOp?!dense_575/StatefulPartitionedCallh
dense_572/CastCastdense_572_input*

DstT0*

SrcT0*'
_output_shapes
:?????????[?
!dense_572/StatefulPartitionedCallStatefulPartitionedCalldense_572/Cast:y:0dense_572_44352659dense_572_44352661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_572_layer_call_and_return_conditional_losses_44352150?
-dense_572/ActivityRegularizer/PartitionedCallPartitionedCall*dense_572/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_572_activity_regularizer_44352093}
#dense_572/ActivityRegularizer/ShapeShape*dense_572/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1dense_572/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_572/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_572/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_572/ActivityRegularizer/strided_sliceStridedSlice,dense_572/ActivityRegularizer/Shape:output:0:dense_572/ActivityRegularizer/strided_slice/stack:output:0<dense_572/ActivityRegularizer/strided_slice/stack_1:output:0<dense_572/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_572/ActivityRegularizer/CastCast4dense_572/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_572/ActivityRegularizer/truedivRealDiv6dense_572/ActivityRegularizer/PartitionedCall:output:0&dense_572/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!dense_573/StatefulPartitionedCallStatefulPartitionedCall*dense_572/StatefulPartitionedCall:output:0dense_573_44352672dense_573_44352674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_573_layer_call_and_return_conditional_losses_44352187?
-dense_573/ActivityRegularizer/PartitionedCallPartitionedCall*dense_573/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_573_activity_regularizer_44352106}
#dense_573/ActivityRegularizer/ShapeShape*dense_573/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1dense_573/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_573/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_573/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_573/ActivityRegularizer/strided_sliceStridedSlice,dense_573/ActivityRegularizer/Shape:output:0:dense_573/ActivityRegularizer/strided_slice/stack:output:0<dense_573/ActivityRegularizer/strided_slice/stack_1:output:0<dense_573/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_573/ActivityRegularizer/CastCast4dense_573/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_573/ActivityRegularizer/truedivRealDiv6dense_573/ActivityRegularizer/PartitionedCall:output:0&dense_573/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!dense_574/StatefulPartitionedCallStatefulPartitionedCall*dense_573/StatefulPartitionedCall:output:0dense_574_44352685dense_574_44352687*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_574_layer_call_and_return_conditional_losses_44352224?
-dense_574/ActivityRegularizer/PartitionedCallPartitionedCall*dense_574/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_574_activity_regularizer_44352119}
#dense_574/ActivityRegularizer/ShapeShape*dense_574/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1dense_574/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_574/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_574/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_574/ActivityRegularizer/strided_sliceStridedSlice,dense_574/ActivityRegularizer/Shape:output:0:dense_574/ActivityRegularizer/strided_slice/stack:output:0<dense_574/ActivityRegularizer/strided_slice/stack_1:output:0<dense_574/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_574/ActivityRegularizer/CastCast4dense_574/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_574/ActivityRegularizer/truedivRealDiv6dense_574/ActivityRegularizer/PartitionedCall:output:0&dense_574/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!dense_575/StatefulPartitionedCallStatefulPartitionedCall*dense_574/StatefulPartitionedCall:output:0dense_575_44352698dense_575_44352700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_575_layer_call_and_return_conditional_losses_44352248?
softmax_143/PartitionedCallPartitionedCall*dense_575/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_softmax_143_layer_call_and_return_conditional_losses_44352259?
2dense_572/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_572_44352659*
_output_shapes

:[
*
dtype0?
#dense_572/kernel/Regularizer/SquareSquare:dense_572/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:[
s
"dense_572/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_572/kernel/Regularizer/SumSum'dense_572/kernel/Regularizer/Square:y:0+dense_572/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_572/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_572/kernel/Regularizer/mulMul+dense_572/kernel/Regularizer/mul/x:output:0)dense_572/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_572/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_572_44352661*
_output_shapes
:
*
dtype0?
!dense_572/bias/Regularizer/SquareSquare8dense_572/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_572/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_572/bias/Regularizer/SumSum%dense_572/bias/Regularizer/Square:y:0)dense_572/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_572/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_572/bias/Regularizer/mulMul)dense_572/bias/Regularizer/mul/x:output:0'dense_572/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_573/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_573_44352672*
_output_shapes

:

*
dtype0?
#dense_573/kernel/Regularizer/SquareSquare:dense_573/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_573/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_573/kernel/Regularizer/SumSum'dense_573/kernel/Regularizer/Square:y:0+dense_573/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_573/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_573/kernel/Regularizer/mulMul+dense_573/kernel/Regularizer/mul/x:output:0)dense_573/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_573/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_573_44352674*
_output_shapes
:
*
dtype0?
!dense_573/bias/Regularizer/SquareSquare8dense_573/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_573/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_573/bias/Regularizer/SumSum%dense_573/bias/Regularizer/Square:y:0)dense_573/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_573/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_573/bias/Regularizer/mulMul)dense_573/bias/Regularizer/mul/x:output:0'dense_573/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_574/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_574_44352685*
_output_shapes

:

*
dtype0?
#dense_574/kernel/Regularizer/SquareSquare:dense_574/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_574/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_574/kernel/Regularizer/SumSum'dense_574/kernel/Regularizer/Square:y:0+dense_574/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_574/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_574/kernel/Regularizer/mulMul+dense_574/kernel/Regularizer/mul/x:output:0)dense_574/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_574/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_574_44352687*
_output_shapes
:
*
dtype0?
!dense_574/bias/Regularizer/SquareSquare8dense_574/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_574/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_574/bias/Regularizer/SumSum%dense_574/bias/Regularizer/Square:y:0)dense_574/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_574/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_574/bias/Regularizer/mulMul)dense_574/bias/Regularizer/mul/x:output:0'dense_574/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
IdentityIdentity$softmax_143/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
i

Identity_1Identity)dense_572/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_2Identity)dense_573/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_3Identity)dense_574/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp"^dense_572/StatefulPartitionedCall1^dense_572/bias/Regularizer/Square/ReadVariableOp3^dense_572/kernel/Regularizer/Square/ReadVariableOp"^dense_573/StatefulPartitionedCall1^dense_573/bias/Regularizer/Square/ReadVariableOp3^dense_573/kernel/Regularizer/Square/ReadVariableOp"^dense_574/StatefulPartitionedCall1^dense_574/bias/Regularizer/Square/ReadVariableOp3^dense_574/kernel/Regularizer/Square/ReadVariableOp"^dense_575/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????[: : : : : : : : 2F
!dense_572/StatefulPartitionedCall!dense_572/StatefulPartitionedCall2d
0dense_572/bias/Regularizer/Square/ReadVariableOp0dense_572/bias/Regularizer/Square/ReadVariableOp2h
2dense_572/kernel/Regularizer/Square/ReadVariableOp2dense_572/kernel/Regularizer/Square/ReadVariableOp2F
!dense_573/StatefulPartitionedCall!dense_573/StatefulPartitionedCall2d
0dense_573/bias/Regularizer/Square/ReadVariableOp0dense_573/bias/Regularizer/Square/ReadVariableOp2h
2dense_573/kernel/Regularizer/Square/ReadVariableOp2dense_573/kernel/Regularizer/Square/ReadVariableOp2F
!dense_574/StatefulPartitionedCall!dense_574/StatefulPartitionedCall2d
0dense_574/bias/Regularizer/Square/ReadVariableOp0dense_574/bias/Regularizer/Square/ReadVariableOp2h
2dense_574/kernel/Regularizer/Square/ReadVariableOp2dense_574/kernel/Regularizer/Square/ReadVariableOp2F
!dense_575/StatefulPartitionedCall!dense_575/StatefulPartitionedCall:X T
'
_output_shapes
:?????????[
)
_user_specified_namedense_572_input
?
?
K__inference_dense_574_layer_call_and_return_all_conditional_losses_44353169

inputs
unknown:


	unknown_0:

identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_574_layer_call_and_return_conditional_losses_44352224?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_574_activity_regularizer_44352119o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?F
?
!__inference__traced_save_44353455
file_prefix/
+savev2_dense_572_kernel_read_readvariableop-
)savev2_dense_572_bias_read_readvariableop/
+savev2_dense_573_kernel_read_readvariableop-
)savev2_dense_573_bias_read_readvariableop/
+savev2_dense_574_kernel_read_readvariableop-
)savev2_dense_574_bias_read_readvariableop/
+savev2_dense_575_kernel_read_readvariableop-
)savev2_dense_575_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_572_kernel_m_read_readvariableop4
0savev2_adam_dense_572_bias_m_read_readvariableop6
2savev2_adam_dense_573_kernel_m_read_readvariableop4
0savev2_adam_dense_573_bias_m_read_readvariableop6
2savev2_adam_dense_574_kernel_m_read_readvariableop4
0savev2_adam_dense_574_bias_m_read_readvariableop6
2savev2_adam_dense_575_kernel_m_read_readvariableop4
0savev2_adam_dense_575_bias_m_read_readvariableop6
2savev2_adam_dense_572_kernel_v_read_readvariableop4
0savev2_adam_dense_572_bias_v_read_readvariableop6
2savev2_adam_dense_573_kernel_v_read_readvariableop4
0savev2_adam_dense_573_bias_v_read_readvariableop6
2savev2_adam_dense_574_kernel_v_read_readvariableop4
0savev2_adam_dense_574_bias_v_read_readvariableop6
2savev2_adam_dense_575_kernel_v_read_readvariableop4
0savev2_adam_dense_575_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_572_kernel_read_readvariableop)savev2_dense_572_bias_read_readvariableop+savev2_dense_573_kernel_read_readvariableop)savev2_dense_573_bias_read_readvariableop+savev2_dense_574_kernel_read_readvariableop)savev2_dense_574_bias_read_readvariableop+savev2_dense_575_kernel_read_readvariableop)savev2_dense_575_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_572_kernel_m_read_readvariableop0savev2_adam_dense_572_bias_m_read_readvariableop2savev2_adam_dense_573_kernel_m_read_readvariableop0savev2_adam_dense_573_bias_m_read_readvariableop2savev2_adam_dense_574_kernel_m_read_readvariableop0savev2_adam_dense_574_bias_m_read_readvariableop2savev2_adam_dense_575_kernel_m_read_readvariableop0savev2_adam_dense_575_bias_m_read_readvariableop2savev2_adam_dense_572_kernel_v_read_readvariableop0savev2_adam_dense_572_bias_v_read_readvariableop2savev2_adam_dense_573_kernel_v_read_readvariableop0savev2_adam_dense_573_bias_v_read_readvariableop2savev2_adam_dense_574_kernel_v_read_readvariableop0savev2_adam_dense_574_bias_v_read_readvariableop2savev2_adam_dense_575_kernel_v_read_readvariableop0savev2_adam_dense_575_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :[
:
:

:
:

:
:

:
: : : : : : : : : :[
:
:

:
:

:
:

:
:[
:
:

:
:

:
:

:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:[
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:[
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:[
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$  

_output_shapes

:

: !

_output_shapes
:
:"

_output_shapes
: 
?
?
__inference_loss_fn_4_44353253M
;dense_574_kernel_regularizer_square_readvariableop_resource:


identity??2dense_574/kernel/Regularizer/Square/ReadVariableOp?
2dense_574/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_574_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:

*
dtype0?
#dense_574/kernel/Regularizer/SquareSquare:dense_574/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_574/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_574/kernel/Regularizer/SumSum'dense_574/kernel/Regularizer/Square:y:0+dense_574/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_574/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_574/kernel/Regularizer/mulMul+dense_574/kernel/Regularizer/mul/x:output:0)dense_574/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_574/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_574/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_574/kernel/Regularizer/Square/ReadVariableOp2dense_574/kernel/Regularizer/Square/ReadVariableOp
?
?
G__inference_dense_573_layer_call_and_return_conditional_losses_44352187

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_573/bias/Regularizer/Square/ReadVariableOp?2dense_573/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
2dense_573/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
#dense_573/kernel/Regularizer/SquareSquare:dense_573/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_573/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_573/kernel/Regularizer/SumSum'dense_573/kernel/Regularizer/Square:y:0+dense_573/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_573/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_573/kernel/Regularizer/mulMul+dense_573/kernel/Regularizer/mul/x:output:0)dense_573/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0dense_573/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_573/bias/Regularizer/SquareSquare8dense_573/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_573/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_573/bias/Regularizer/SumSum%dense_573/bias/Regularizer/Square:y:0)dense_573/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_573/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_573/bias/Regularizer/mulMul)dense_573/bias/Regularizer/mul/x:output:0'dense_573/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_573/bias/Regularizer/Square/ReadVariableOp3^dense_573/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_573/bias/Regularizer/Square/ReadVariableOp0dense_573/bias/Regularizer/Square/ReadVariableOp2h
2dense_573/kernel/Regularizer/Square/ReadVariableOp2dense_573/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
__inference_loss_fn_5_44353264G
9dense_574_bias_regularizer_square_readvariableop_resource:

identity??0dense_574/bias/Regularizer/Square/ReadVariableOp?
0dense_574/bias/Regularizer/Square/ReadVariableOpReadVariableOp9dense_574_bias_regularizer_square_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_574/bias/Regularizer/SquareSquare8dense_574/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_574/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_574/bias/Regularizer/SumSum%dense_574/bias/Regularizer/Square:y:0)dense_574/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_574/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_574/bias/Regularizer/mulMul)dense_574/bias/Regularizer/mul/x:output:0'dense_574/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_574/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^dense_574/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_574/bias/Regularizer/Square/ReadVariableOp0dense_574/bias/Regularizer/Square/ReadVariableOp
?	
?
&__inference_signature_wrapper_44352809
dense_572_input
unknown:[

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:


	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_572_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_44352080o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????[: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????[
)
_user_specified_namedense_572_input
?	
?
G__inference_dense_575_layer_call_and_return_conditional_losses_44353188

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
1__inference_sequential_143_layer_call_fn_44352833

inputs
unknown:[

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:


	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????
: : : **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352301o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????[: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????[
 
_user_specified_nameinputs
?d
?
#__inference__wrapped_model_44352080
dense_572_inputI
7sequential_143_dense_572_matmul_readvariableop_resource:[
F
8sequential_143_dense_572_biasadd_readvariableop_resource:
I
7sequential_143_dense_573_matmul_readvariableop_resource:

F
8sequential_143_dense_573_biasadd_readvariableop_resource:
I
7sequential_143_dense_574_matmul_readvariableop_resource:

F
8sequential_143_dense_574_biasadd_readvariableop_resource:
I
7sequential_143_dense_575_matmul_readvariableop_resource:

F
8sequential_143_dense_575_biasadd_readvariableop_resource:

identity??/sequential_143/dense_572/BiasAdd/ReadVariableOp?.sequential_143/dense_572/MatMul/ReadVariableOp?/sequential_143/dense_573/BiasAdd/ReadVariableOp?.sequential_143/dense_573/MatMul/ReadVariableOp?/sequential_143/dense_574/BiasAdd/ReadVariableOp?.sequential_143/dense_574/MatMul/ReadVariableOp?/sequential_143/dense_575/BiasAdd/ReadVariableOp?.sequential_143/dense_575/MatMul/ReadVariableOpw
sequential_143/dense_572/CastCastdense_572_input*

DstT0*

SrcT0*'
_output_shapes
:?????????[?
.sequential_143/dense_572/MatMul/ReadVariableOpReadVariableOp7sequential_143_dense_572_matmul_readvariableop_resource*
_output_shapes

:[
*
dtype0?
sequential_143/dense_572/MatMulMatMul!sequential_143/dense_572/Cast:y:06sequential_143/dense_572/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
/sequential_143/dense_572/BiasAdd/ReadVariableOpReadVariableOp8sequential_143_dense_572_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
 sequential_143/dense_572/BiasAddBiasAdd)sequential_143/dense_572/MatMul:product:07sequential_143/dense_572/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
sequential_143/dense_572/ReluRelu)sequential_143/dense_572/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
3sequential_143/dense_572/ActivityRegularizer/SquareSquare+sequential_143/dense_572/Relu:activations:0*
T0*'
_output_shapes
:?????????
?
2sequential_143/dense_572/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
0sequential_143/dense_572/ActivityRegularizer/SumSum7sequential_143/dense_572/ActivityRegularizer/Square:y:0;sequential_143/dense_572/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: w
2sequential_143/dense_572/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
0sequential_143/dense_572/ActivityRegularizer/mulMul;sequential_143/dense_572/ActivityRegularizer/mul/x:output:09sequential_143/dense_572/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
2sequential_143/dense_572/ActivityRegularizer/ShapeShape+sequential_143/dense_572/Relu:activations:0*
T0*
_output_shapes
:?
@sequential_143/dense_572/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsequential_143/dense_572/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential_143/dense_572/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential_143/dense_572/ActivityRegularizer/strided_sliceStridedSlice;sequential_143/dense_572/ActivityRegularizer/Shape:output:0Isequential_143/dense_572/ActivityRegularizer/strided_slice/stack:output:0Ksequential_143/dense_572/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_143/dense_572/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1sequential_143/dense_572/ActivityRegularizer/CastCastCsequential_143/dense_572/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
4sequential_143/dense_572/ActivityRegularizer/truedivRealDiv4sequential_143/dense_572/ActivityRegularizer/mul:z:05sequential_143/dense_572/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
.sequential_143/dense_573/MatMul/ReadVariableOpReadVariableOp7sequential_143_dense_573_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
sequential_143/dense_573/MatMulMatMul+sequential_143/dense_572/Relu:activations:06sequential_143/dense_573/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
/sequential_143/dense_573/BiasAdd/ReadVariableOpReadVariableOp8sequential_143_dense_573_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
 sequential_143/dense_573/BiasAddBiasAdd)sequential_143/dense_573/MatMul:product:07sequential_143/dense_573/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
sequential_143/dense_573/ReluRelu)sequential_143/dense_573/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
3sequential_143/dense_573/ActivityRegularizer/SquareSquare+sequential_143/dense_573/Relu:activations:0*
T0*'
_output_shapes
:?????????
?
2sequential_143/dense_573/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
0sequential_143/dense_573/ActivityRegularizer/SumSum7sequential_143/dense_573/ActivityRegularizer/Square:y:0;sequential_143/dense_573/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: w
2sequential_143/dense_573/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
0sequential_143/dense_573/ActivityRegularizer/mulMul;sequential_143/dense_573/ActivityRegularizer/mul/x:output:09sequential_143/dense_573/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
2sequential_143/dense_573/ActivityRegularizer/ShapeShape+sequential_143/dense_573/Relu:activations:0*
T0*
_output_shapes
:?
@sequential_143/dense_573/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsequential_143/dense_573/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential_143/dense_573/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential_143/dense_573/ActivityRegularizer/strided_sliceStridedSlice;sequential_143/dense_573/ActivityRegularizer/Shape:output:0Isequential_143/dense_573/ActivityRegularizer/strided_slice/stack:output:0Ksequential_143/dense_573/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_143/dense_573/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1sequential_143/dense_573/ActivityRegularizer/CastCastCsequential_143/dense_573/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
4sequential_143/dense_573/ActivityRegularizer/truedivRealDiv4sequential_143/dense_573/ActivityRegularizer/mul:z:05sequential_143/dense_573/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
.sequential_143/dense_574/MatMul/ReadVariableOpReadVariableOp7sequential_143_dense_574_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
sequential_143/dense_574/MatMulMatMul+sequential_143/dense_573/Relu:activations:06sequential_143/dense_574/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
/sequential_143/dense_574/BiasAdd/ReadVariableOpReadVariableOp8sequential_143_dense_574_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
 sequential_143/dense_574/BiasAddBiasAdd)sequential_143/dense_574/MatMul:product:07sequential_143/dense_574/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
sequential_143/dense_574/ReluRelu)sequential_143/dense_574/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
3sequential_143/dense_574/ActivityRegularizer/SquareSquare+sequential_143/dense_574/Relu:activations:0*
T0*'
_output_shapes
:?????????
?
2sequential_143/dense_574/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
0sequential_143/dense_574/ActivityRegularizer/SumSum7sequential_143/dense_574/ActivityRegularizer/Square:y:0;sequential_143/dense_574/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: w
2sequential_143/dense_574/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
0sequential_143/dense_574/ActivityRegularizer/mulMul;sequential_143/dense_574/ActivityRegularizer/mul/x:output:09sequential_143/dense_574/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
2sequential_143/dense_574/ActivityRegularizer/ShapeShape+sequential_143/dense_574/Relu:activations:0*
T0*
_output_shapes
:?
@sequential_143/dense_574/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsequential_143/dense_574/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential_143/dense_574/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential_143/dense_574/ActivityRegularizer/strided_sliceStridedSlice;sequential_143/dense_574/ActivityRegularizer/Shape:output:0Isequential_143/dense_574/ActivityRegularizer/strided_slice/stack:output:0Ksequential_143/dense_574/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_143/dense_574/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1sequential_143/dense_574/ActivityRegularizer/CastCastCsequential_143/dense_574/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
4sequential_143/dense_574/ActivityRegularizer/truedivRealDiv4sequential_143/dense_574/ActivityRegularizer/mul:z:05sequential_143/dense_574/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
.sequential_143/dense_575/MatMul/ReadVariableOpReadVariableOp7sequential_143_dense_575_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
sequential_143/dense_575/MatMulMatMul+sequential_143/dense_574/Relu:activations:06sequential_143/dense_575/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
/sequential_143/dense_575/BiasAdd/ReadVariableOpReadVariableOp8sequential_143_dense_575_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
 sequential_143/dense_575/BiasAddBiasAdd)sequential_143/dense_575/MatMul:product:07sequential_143/dense_575/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
"sequential_143/softmax_143/SoftmaxSoftmax)sequential_143/dense_575/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
{
IdentityIdentity,sequential_143/softmax_143/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp0^sequential_143/dense_572/BiasAdd/ReadVariableOp/^sequential_143/dense_572/MatMul/ReadVariableOp0^sequential_143/dense_573/BiasAdd/ReadVariableOp/^sequential_143/dense_573/MatMul/ReadVariableOp0^sequential_143/dense_574/BiasAdd/ReadVariableOp/^sequential_143/dense_574/MatMul/ReadVariableOp0^sequential_143/dense_575/BiasAdd/ReadVariableOp/^sequential_143/dense_575/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????[: : : : : : : : 2b
/sequential_143/dense_572/BiasAdd/ReadVariableOp/sequential_143/dense_572/BiasAdd/ReadVariableOp2`
.sequential_143/dense_572/MatMul/ReadVariableOp.sequential_143/dense_572/MatMul/ReadVariableOp2b
/sequential_143/dense_573/BiasAdd/ReadVariableOp/sequential_143/dense_573/BiasAdd/ReadVariableOp2`
.sequential_143/dense_573/MatMul/ReadVariableOp.sequential_143/dense_573/MatMul/ReadVariableOp2b
/sequential_143/dense_574/BiasAdd/ReadVariableOp/sequential_143/dense_574/BiasAdd/ReadVariableOp2`
.sequential_143/dense_574/MatMul/ReadVariableOp.sequential_143/dense_574/MatMul/ReadVariableOp2b
/sequential_143/dense_575/BiasAdd/ReadVariableOp/sequential_143/dense_575/BiasAdd/ReadVariableOp2`
.sequential_143/dense_575/MatMul/ReadVariableOp.sequential_143/dense_575/MatMul/ReadVariableOp:X T
'
_output_shapes
:?????????[
)
_user_specified_namedense_572_input
?	
?
1__inference_sequential_143_layer_call_fn_44352323
dense_572_input
unknown:[

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:


	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_572_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????
: : : **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352301o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????[: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????[
)
_user_specified_namedense_572_input
?	
?
G__inference_dense_575_layer_call_and_return_conditional_losses_44352248

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
K__inference_dense_572_layer_call_and_return_all_conditional_losses_44353105

inputs
unknown:[

	unknown_0:

identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_572_layer_call_and_return_conditional_losses_44352150?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_572_activity_regularizer_44352093o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????[: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????[
 
_user_specified_nameinputs
?
J
.__inference_softmax_143_layer_call_fn_44353193

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_softmax_143_layer_call_and_return_conditional_losses_44352259`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
G__inference_dense_574_layer_call_and_return_conditional_losses_44352224

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_574/bias/Regularizer/Square/ReadVariableOp?2dense_574/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
2dense_574/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
#dense_574/kernel/Regularizer/SquareSquare:dense_574/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_574/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_574/kernel/Regularizer/SumSum'dense_574/kernel/Regularizer/Square:y:0+dense_574/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_574/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_574/kernel/Regularizer/mulMul+dense_574/kernel/Regularizer/mul/x:output:0)dense_574/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0dense_574/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_574/bias/Regularizer/SquareSquare8dense_574/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_574/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_574/bias/Regularizer/SumSum%dense_574/bias/Regularizer/Square:y:0)dense_574/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_574/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_574/bias/Regularizer/mulMul)dense_574/bias/Regularizer/mul/x:output:0'dense_574/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_574/bias/Regularizer/Square/ReadVariableOp3^dense_574/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_574/bias/Regularizer/Square/ReadVariableOp0dense_574/bias/Regularizer/Square/ReadVariableOp2h
2dense_574/kernel/Regularizer/Square/ReadVariableOp2dense_574/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_44353231M
;dense_573_kernel_regularizer_square_readvariableop_resource:


identity??2dense_573/kernel/Regularizer/Square/ReadVariableOp?
2dense_573/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_573_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:

*
dtype0?
#dense_573/kernel/Regularizer/SquareSquare:dense_573/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_573/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_573/kernel/Regularizer/SumSum'dense_573/kernel/Regularizer/Square:y:0+dense_573/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_573/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_573/kernel/Regularizer/mulMul+dense_573/kernel/Regularizer/mul/x:output:0)dense_573/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_573/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_573/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_573/kernel/Regularizer/Square/ReadVariableOp2dense_573/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_3_44353242G
9dense_573_bias_regularizer_square_readvariableop_resource:

identity??0dense_573/bias/Regularizer/Square/ReadVariableOp?
0dense_573/bias/Regularizer/Square/ReadVariableOpReadVariableOp9dense_573_bias_regularizer_square_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_573/bias/Regularizer/SquareSquare8dense_573/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_573/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_573/bias/Regularizer/SumSum%dense_573/bias/Regularizer/Square:y:0)dense_573/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_573/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_573/bias/Regularizer/mulMul)dense_573/bias/Regularizer/mul/x:output:0'dense_573/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_573/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^dense_573/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_573/bias/Regularizer/Square/ReadVariableOp0dense_573/bias/Regularizer/Square/ReadVariableOp
?
?
G__inference_dense_572_layer_call_and_return_conditional_losses_44352150

inputs0
matmul_readvariableop_resource:[
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_572/bias/Regularizer/Square/ReadVariableOp?2dense_572/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:[
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
2dense_572/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:[
*
dtype0?
#dense_572/kernel/Regularizer/SquareSquare:dense_572/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:[
s
"dense_572/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_572/kernel/Regularizer/SumSum'dense_572/kernel/Regularizer/Square:y:0+dense_572/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_572/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_572/kernel/Regularizer/mulMul+dense_572/kernel/Regularizer/mul/x:output:0)dense_572/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0dense_572/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_572/bias/Regularizer/SquareSquare8dense_572/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_572/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_572/bias/Regularizer/SumSum%dense_572/bias/Regularizer/Square:y:0)dense_572/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_572/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_572/bias/Regularizer/mulMul)dense_572/bias/Regularizer/mul/x:output:0'dense_572/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_572/bias/Regularizer/Square/ReadVariableOp3^dense_572/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????[: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_572/bias/Regularizer/Square/ReadVariableOp0dense_572/bias/Regularizer/Square/ReadVariableOp2h
2dense_572/kernel/Regularizer/Square/ReadVariableOp2dense_572/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????[
 
_user_specified_nameinputs
?r
?
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352655
dense_572_input$
dense_572_44352570:[
 
dense_572_44352572:
$
dense_573_44352583:

 
dense_573_44352585:
$
dense_574_44352596:

 
dense_574_44352598:
$
dense_575_44352609:

 
dense_575_44352611:

identity

identity_1

identity_2

identity_3??!dense_572/StatefulPartitionedCall?0dense_572/bias/Regularizer/Square/ReadVariableOp?2dense_572/kernel/Regularizer/Square/ReadVariableOp?!dense_573/StatefulPartitionedCall?0dense_573/bias/Regularizer/Square/ReadVariableOp?2dense_573/kernel/Regularizer/Square/ReadVariableOp?!dense_574/StatefulPartitionedCall?0dense_574/bias/Regularizer/Square/ReadVariableOp?2dense_574/kernel/Regularizer/Square/ReadVariableOp?!dense_575/StatefulPartitionedCallh
dense_572/CastCastdense_572_input*

DstT0*

SrcT0*'
_output_shapes
:?????????[?
!dense_572/StatefulPartitionedCallStatefulPartitionedCalldense_572/Cast:y:0dense_572_44352570dense_572_44352572*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_572_layer_call_and_return_conditional_losses_44352150?
-dense_572/ActivityRegularizer/PartitionedCallPartitionedCall*dense_572/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_572_activity_regularizer_44352093}
#dense_572/ActivityRegularizer/ShapeShape*dense_572/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1dense_572/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_572/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_572/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_572/ActivityRegularizer/strided_sliceStridedSlice,dense_572/ActivityRegularizer/Shape:output:0:dense_572/ActivityRegularizer/strided_slice/stack:output:0<dense_572/ActivityRegularizer/strided_slice/stack_1:output:0<dense_572/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_572/ActivityRegularizer/CastCast4dense_572/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_572/ActivityRegularizer/truedivRealDiv6dense_572/ActivityRegularizer/PartitionedCall:output:0&dense_572/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!dense_573/StatefulPartitionedCallStatefulPartitionedCall*dense_572/StatefulPartitionedCall:output:0dense_573_44352583dense_573_44352585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_573_layer_call_and_return_conditional_losses_44352187?
-dense_573/ActivityRegularizer/PartitionedCallPartitionedCall*dense_573/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_573_activity_regularizer_44352106}
#dense_573/ActivityRegularizer/ShapeShape*dense_573/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1dense_573/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_573/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_573/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_573/ActivityRegularizer/strided_sliceStridedSlice,dense_573/ActivityRegularizer/Shape:output:0:dense_573/ActivityRegularizer/strided_slice/stack:output:0<dense_573/ActivityRegularizer/strided_slice/stack_1:output:0<dense_573/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_573/ActivityRegularizer/CastCast4dense_573/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_573/ActivityRegularizer/truedivRealDiv6dense_573/ActivityRegularizer/PartitionedCall:output:0&dense_573/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!dense_574/StatefulPartitionedCallStatefulPartitionedCall*dense_573/StatefulPartitionedCall:output:0dense_574_44352596dense_574_44352598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_574_layer_call_and_return_conditional_losses_44352224?
-dense_574/ActivityRegularizer/PartitionedCallPartitionedCall*dense_574/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_574_activity_regularizer_44352119}
#dense_574/ActivityRegularizer/ShapeShape*dense_574/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1dense_574/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_574/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_574/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_574/ActivityRegularizer/strided_sliceStridedSlice,dense_574/ActivityRegularizer/Shape:output:0:dense_574/ActivityRegularizer/strided_slice/stack:output:0<dense_574/ActivityRegularizer/strided_slice/stack_1:output:0<dense_574/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_574/ActivityRegularizer/CastCast4dense_574/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_574/ActivityRegularizer/truedivRealDiv6dense_574/ActivityRegularizer/PartitionedCall:output:0&dense_574/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!dense_575/StatefulPartitionedCallStatefulPartitionedCall*dense_574/StatefulPartitionedCall:output:0dense_575_44352609dense_575_44352611*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_575_layer_call_and_return_conditional_losses_44352248?
softmax_143/PartitionedCallPartitionedCall*dense_575/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_softmax_143_layer_call_and_return_conditional_losses_44352259?
2dense_572/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_572_44352570*
_output_shapes

:[
*
dtype0?
#dense_572/kernel/Regularizer/SquareSquare:dense_572/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:[
s
"dense_572/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_572/kernel/Regularizer/SumSum'dense_572/kernel/Regularizer/Square:y:0+dense_572/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_572/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_572/kernel/Regularizer/mulMul+dense_572/kernel/Regularizer/mul/x:output:0)dense_572/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_572/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_572_44352572*
_output_shapes
:
*
dtype0?
!dense_572/bias/Regularizer/SquareSquare8dense_572/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_572/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_572/bias/Regularizer/SumSum%dense_572/bias/Regularizer/Square:y:0)dense_572/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_572/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_572/bias/Regularizer/mulMul)dense_572/bias/Regularizer/mul/x:output:0'dense_572/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_573/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_573_44352583*
_output_shapes

:

*
dtype0?
#dense_573/kernel/Regularizer/SquareSquare:dense_573/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_573/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_573/kernel/Regularizer/SumSum'dense_573/kernel/Regularizer/Square:y:0+dense_573/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_573/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_573/kernel/Regularizer/mulMul+dense_573/kernel/Regularizer/mul/x:output:0)dense_573/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_573/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_573_44352585*
_output_shapes
:
*
dtype0?
!dense_573/bias/Regularizer/SquareSquare8dense_573/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_573/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_573/bias/Regularizer/SumSum%dense_573/bias/Regularizer/Square:y:0)dense_573/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_573/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_573/bias/Regularizer/mulMul)dense_573/bias/Regularizer/mul/x:output:0'dense_573/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_574/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_574_44352596*
_output_shapes

:

*
dtype0?
#dense_574/kernel/Regularizer/SquareSquare:dense_574/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_574/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_574/kernel/Regularizer/SumSum'dense_574/kernel/Regularizer/Square:y:0+dense_574/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_574/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_574/kernel/Regularizer/mulMul+dense_574/kernel/Regularizer/mul/x:output:0)dense_574/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_574/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_574_44352598*
_output_shapes
:
*
dtype0?
!dense_574/bias/Regularizer/SquareSquare8dense_574/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_574/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_574/bias/Regularizer/SumSum%dense_574/bias/Regularizer/Square:y:0)dense_574/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_574/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_574/bias/Regularizer/mulMul)dense_574/bias/Regularizer/mul/x:output:0'dense_574/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
IdentityIdentity$softmax_143/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
i

Identity_1Identity)dense_572/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_2Identity)dense_573/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_3Identity)dense_574/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp"^dense_572/StatefulPartitionedCall1^dense_572/bias/Regularizer/Square/ReadVariableOp3^dense_572/kernel/Regularizer/Square/ReadVariableOp"^dense_573/StatefulPartitionedCall1^dense_573/bias/Regularizer/Square/ReadVariableOp3^dense_573/kernel/Regularizer/Square/ReadVariableOp"^dense_574/StatefulPartitionedCall1^dense_574/bias/Regularizer/Square/ReadVariableOp3^dense_574/kernel/Regularizer/Square/ReadVariableOp"^dense_575/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????[: : : : : : : : 2F
!dense_572/StatefulPartitionedCall!dense_572/StatefulPartitionedCall2d
0dense_572/bias/Regularizer/Square/ReadVariableOp0dense_572/bias/Regularizer/Square/ReadVariableOp2h
2dense_572/kernel/Regularizer/Square/ReadVariableOp2dense_572/kernel/Regularizer/Square/ReadVariableOp2F
!dense_573/StatefulPartitionedCall!dense_573/StatefulPartitionedCall2d
0dense_573/bias/Regularizer/Square/ReadVariableOp0dense_573/bias/Regularizer/Square/ReadVariableOp2h
2dense_573/kernel/Regularizer/Square/ReadVariableOp2dense_573/kernel/Regularizer/Square/ReadVariableOp2F
!dense_574/StatefulPartitionedCall!dense_574/StatefulPartitionedCall2d
0dense_574/bias/Regularizer/Square/ReadVariableOp0dense_574/bias/Regularizer/Square/ReadVariableOp2h
2dense_574/kernel/Regularizer/Square/ReadVariableOp2dense_574/kernel/Regularizer/Square/ReadVariableOp2F
!dense_575/StatefulPartitionedCall!dense_575/StatefulPartitionedCall:X T
'
_output_shapes
:?????????[
)
_user_specified_namedense_572_input
??
?
$__inference__traced_restore_44353564
file_prefix3
!assignvariableop_dense_572_kernel:[
/
!assignvariableop_1_dense_572_bias:
5
#assignvariableop_2_dense_573_kernel:

/
!assignvariableop_3_dense_573_bias:
5
#assignvariableop_4_dense_574_kernel:

/
!assignvariableop_5_dense_574_bias:
5
#assignvariableop_6_dense_575_kernel:

/
!assignvariableop_7_dense_575_bias:
&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: =
+assignvariableop_17_adam_dense_572_kernel_m:[
7
)assignvariableop_18_adam_dense_572_bias_m:
=
+assignvariableop_19_adam_dense_573_kernel_m:

7
)assignvariableop_20_adam_dense_573_bias_m:
=
+assignvariableop_21_adam_dense_574_kernel_m:

7
)assignvariableop_22_adam_dense_574_bias_m:
=
+assignvariableop_23_adam_dense_575_kernel_m:

7
)assignvariableop_24_adam_dense_575_bias_m:
=
+assignvariableop_25_adam_dense_572_kernel_v:[
7
)assignvariableop_26_adam_dense_572_bias_v:
=
+assignvariableop_27_adam_dense_573_kernel_v:

7
)assignvariableop_28_adam_dense_573_bias_v:
=
+assignvariableop_29_adam_dense_574_kernel_v:

7
)assignvariableop_30_adam_dense_574_bias_v:
=
+assignvariableop_31_adam_dense_575_kernel_v:

7
)assignvariableop_32_adam_dense_575_bias_v:

identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_dense_572_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_572_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_573_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_573_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_574_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_574_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_575_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_575_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_dense_572_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_572_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_573_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_573_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_574_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_574_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_575_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_575_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_572_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_572_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_573_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_573_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_574_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_574_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_575_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_575_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
,__inference_dense_574_layer_call_fn_44353158

inputs
unknown:


	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_574_layer_call_and_return_conditional_losses_44352224o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
??
?	
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352965

inputs:
(dense_572_matmul_readvariableop_resource:[
7
)dense_572_biasadd_readvariableop_resource:
:
(dense_573_matmul_readvariableop_resource:

7
)dense_573_biasadd_readvariableop_resource:
:
(dense_574_matmul_readvariableop_resource:

7
)dense_574_biasadd_readvariableop_resource:
:
(dense_575_matmul_readvariableop_resource:

7
)dense_575_biasadd_readvariableop_resource:

identity

identity_1

identity_2

identity_3?? dense_572/BiasAdd/ReadVariableOp?dense_572/MatMul/ReadVariableOp?0dense_572/bias/Regularizer/Square/ReadVariableOp?2dense_572/kernel/Regularizer/Square/ReadVariableOp? dense_573/BiasAdd/ReadVariableOp?dense_573/MatMul/ReadVariableOp?0dense_573/bias/Regularizer/Square/ReadVariableOp?2dense_573/kernel/Regularizer/Square/ReadVariableOp? dense_574/BiasAdd/ReadVariableOp?dense_574/MatMul/ReadVariableOp?0dense_574/bias/Regularizer/Square/ReadVariableOp?2dense_574/kernel/Regularizer/Square/ReadVariableOp? dense_575/BiasAdd/ReadVariableOp?dense_575/MatMul/ReadVariableOp_
dense_572/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????[?
dense_572/MatMul/ReadVariableOpReadVariableOp(dense_572_matmul_readvariableop_resource*
_output_shapes

:[
*
dtype0?
dense_572/MatMulMatMuldense_572/Cast:y:0'dense_572/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 dense_572/BiasAdd/ReadVariableOpReadVariableOp)dense_572_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_572/BiasAddBiasAdddense_572/MatMul:product:0(dense_572/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
d
dense_572/ReluReludense_572/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
~
$dense_572/ActivityRegularizer/SquareSquaredense_572/Relu:activations:0*
T0*'
_output_shapes
:?????????
t
#dense_572/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_572/ActivityRegularizer/SumSum(dense_572/ActivityRegularizer/Square:y:0,dense_572/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#dense_572/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!dense_572/ActivityRegularizer/mulMul,dense_572/ActivityRegularizer/mul/x:output:0*dense_572/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: o
#dense_572/ActivityRegularizer/ShapeShapedense_572/Relu:activations:0*
T0*
_output_shapes
:{
1dense_572/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_572/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_572/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_572/ActivityRegularizer/strided_sliceStridedSlice,dense_572/ActivityRegularizer/Shape:output:0:dense_572/ActivityRegularizer/strided_slice/stack:output:0<dense_572/ActivityRegularizer/strided_slice/stack_1:output:0<dense_572/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_572/ActivityRegularizer/CastCast4dense_572/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_572/ActivityRegularizer/truedivRealDiv%dense_572/ActivityRegularizer/mul:z:0&dense_572/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
dense_573/MatMul/ReadVariableOpReadVariableOp(dense_573_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
dense_573/MatMulMatMuldense_572/Relu:activations:0'dense_573/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 dense_573/BiasAdd/ReadVariableOpReadVariableOp)dense_573_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_573/BiasAddBiasAdddense_573/MatMul:product:0(dense_573/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
d
dense_573/ReluReludense_573/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
~
$dense_573/ActivityRegularizer/SquareSquaredense_573/Relu:activations:0*
T0*'
_output_shapes
:?????????
t
#dense_573/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_573/ActivityRegularizer/SumSum(dense_573/ActivityRegularizer/Square:y:0,dense_573/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#dense_573/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!dense_573/ActivityRegularizer/mulMul,dense_573/ActivityRegularizer/mul/x:output:0*dense_573/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: o
#dense_573/ActivityRegularizer/ShapeShapedense_573/Relu:activations:0*
T0*
_output_shapes
:{
1dense_573/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_573/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_573/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_573/ActivityRegularizer/strided_sliceStridedSlice,dense_573/ActivityRegularizer/Shape:output:0:dense_573/ActivityRegularizer/strided_slice/stack:output:0<dense_573/ActivityRegularizer/strided_slice/stack_1:output:0<dense_573/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_573/ActivityRegularizer/CastCast4dense_573/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_573/ActivityRegularizer/truedivRealDiv%dense_573/ActivityRegularizer/mul:z:0&dense_573/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
dense_574/MatMul/ReadVariableOpReadVariableOp(dense_574_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
dense_574/MatMulMatMuldense_573/Relu:activations:0'dense_574/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 dense_574/BiasAdd/ReadVariableOpReadVariableOp)dense_574_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_574/BiasAddBiasAdddense_574/MatMul:product:0(dense_574/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
d
dense_574/ReluReludense_574/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
~
$dense_574/ActivityRegularizer/SquareSquaredense_574/Relu:activations:0*
T0*'
_output_shapes
:?????????
t
#dense_574/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_574/ActivityRegularizer/SumSum(dense_574/ActivityRegularizer/Square:y:0,dense_574/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#dense_574/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!dense_574/ActivityRegularizer/mulMul,dense_574/ActivityRegularizer/mul/x:output:0*dense_574/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: o
#dense_574/ActivityRegularizer/ShapeShapedense_574/Relu:activations:0*
T0*
_output_shapes
:{
1dense_574/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_574/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_574/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_574/ActivityRegularizer/strided_sliceStridedSlice,dense_574/ActivityRegularizer/Shape:output:0:dense_574/ActivityRegularizer/strided_slice/stack:output:0<dense_574/ActivityRegularizer/strided_slice/stack_1:output:0<dense_574/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"dense_574/ActivityRegularizer/CastCast4dense_574/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%dense_574/ActivityRegularizer/truedivRealDiv%dense_574/ActivityRegularizer/mul:z:0&dense_574/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
dense_575/MatMul/ReadVariableOpReadVariableOp(dense_575_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
dense_575/MatMulMatMuldense_574/Relu:activations:0'dense_575/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 dense_575/BiasAdd/ReadVariableOpReadVariableOp)dense_575_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_575/BiasAddBiasAdddense_575/MatMul:product:0(dense_575/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
l
softmax_143/SoftmaxSoftmaxdense_575/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
2dense_572/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_572_matmul_readvariableop_resource*
_output_shapes

:[
*
dtype0?
#dense_572/kernel/Regularizer/SquareSquare:dense_572/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:[
s
"dense_572/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_572/kernel/Regularizer/SumSum'dense_572/kernel/Regularizer/Square:y:0+dense_572/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_572/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_572/kernel/Regularizer/mulMul+dense_572/kernel/Regularizer/mul/x:output:0)dense_572/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0dense_572/bias/Regularizer/Square/ReadVariableOpReadVariableOp)dense_572_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_572/bias/Regularizer/SquareSquare8dense_572/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_572/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_572/bias/Regularizer/SumSum%dense_572/bias/Regularizer/Square:y:0)dense_572/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_572/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_572/bias/Regularizer/mulMul)dense_572/bias/Regularizer/mul/x:output:0'dense_572/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_573/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_573_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
#dense_573/kernel/Regularizer/SquareSquare:dense_573/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_573/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_573/kernel/Regularizer/SumSum'dense_573/kernel/Regularizer/Square:y:0+dense_573/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_573/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_573/kernel/Regularizer/mulMul+dense_573/kernel/Regularizer/mul/x:output:0)dense_573/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0dense_573/bias/Regularizer/Square/ReadVariableOpReadVariableOp)dense_573_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_573/bias/Regularizer/SquareSquare8dense_573/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_573/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_573/bias/Regularizer/SumSum%dense_573/bias/Regularizer/Square:y:0)dense_573/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_573/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_573/bias/Regularizer/mulMul)dense_573/bias/Regularizer/mul/x:output:0'dense_573/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
2dense_574/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_574_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0?
#dense_574/kernel/Regularizer/SquareSquare:dense_574/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:

s
"dense_574/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_574/kernel/Regularizer/SumSum'dense_574/kernel/Regularizer/Square:y:0+dense_574/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_574/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
 dense_574/kernel/Regularizer/mulMul+dense_574/kernel/Regularizer/mul/x:output:0)dense_574/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0dense_574/bias/Regularizer/Square/ReadVariableOpReadVariableOp)dense_574_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
!dense_574/bias/Regularizer/SquareSquare8dense_574/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
j
 dense_574/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_574/bias/Regularizer/SumSum%dense_574/bias/Regularizer/Square:y:0)dense_574/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_574/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
dense_574/bias/Regularizer/mulMul)dense_574/bias/Regularizer/mul/x:output:0'dense_574/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: l
IdentityIdentitysoftmax_143/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
i

Identity_1Identity)dense_572/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_2Identity)dense_573/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_3Identity)dense_574/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp!^dense_572/BiasAdd/ReadVariableOp ^dense_572/MatMul/ReadVariableOp1^dense_572/bias/Regularizer/Square/ReadVariableOp3^dense_572/kernel/Regularizer/Square/ReadVariableOp!^dense_573/BiasAdd/ReadVariableOp ^dense_573/MatMul/ReadVariableOp1^dense_573/bias/Regularizer/Square/ReadVariableOp3^dense_573/kernel/Regularizer/Square/ReadVariableOp!^dense_574/BiasAdd/ReadVariableOp ^dense_574/MatMul/ReadVariableOp1^dense_574/bias/Regularizer/Square/ReadVariableOp3^dense_574/kernel/Regularizer/Square/ReadVariableOp!^dense_575/BiasAdd/ReadVariableOp ^dense_575/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????[: : : : : : : : 2D
 dense_572/BiasAdd/ReadVariableOp dense_572/BiasAdd/ReadVariableOp2B
dense_572/MatMul/ReadVariableOpdense_572/MatMul/ReadVariableOp2d
0dense_572/bias/Regularizer/Square/ReadVariableOp0dense_572/bias/Regularizer/Square/ReadVariableOp2h
2dense_572/kernel/Regularizer/Square/ReadVariableOp2dense_572/kernel/Regularizer/Square/ReadVariableOp2D
 dense_573/BiasAdd/ReadVariableOp dense_573/BiasAdd/ReadVariableOp2B
dense_573/MatMul/ReadVariableOpdense_573/MatMul/ReadVariableOp2d
0dense_573/bias/Regularizer/Square/ReadVariableOp0dense_573/bias/Regularizer/Square/ReadVariableOp2h
2dense_573/kernel/Regularizer/Square/ReadVariableOp2dense_573/kernel/Regularizer/Square/ReadVariableOp2D
 dense_574/BiasAdd/ReadVariableOp dense_574/BiasAdd/ReadVariableOp2B
dense_574/MatMul/ReadVariableOpdense_574/MatMul/ReadVariableOp2d
0dense_574/bias/Regularizer/Square/ReadVariableOp0dense_574/bias/Regularizer/Square/ReadVariableOp2h
2dense_574/kernel/Regularizer/Square/ReadVariableOp2dense_574/kernel/Regularizer/Square/ReadVariableOp2D
 dense_575/BiasAdd/ReadVariableOp dense_575/BiasAdd/ReadVariableOp2B
dense_575/MatMul/ReadVariableOpdense_575/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????[
 
_user_specified_nameinputs
?
J
3__inference_dense_572_activity_regularizer_44352093
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????G
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
?	
?
1__inference_sequential_143_layer_call_fn_44352566
dense_572_input
unknown:[

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:


	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_572_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????
: : : **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352520o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????[: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????[
)
_user_specified_namedense_572_input
?
e
I__inference_softmax_143_layer_call_and_return_conditional_losses_44352259

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
,__inference_dense_575_layer_call_fn_44353178

inputs
unknown:


	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_575_layer_call_and_return_conditional_losses_44352248o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
,__inference_dense_572_layer_call_fn_44353094

inputs
unknown:[

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_572_layer_call_and_return_conditional_losses_44352150o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????[: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????[
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
dense_572_input8
!serving_default_dense_572_input:0?????????[?
softmax_1430
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
f__call__
*g&call_and_return_all_conditional_losses
h_default_save_signature"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
?
(iter

)beta_1

*beta_2
	+decay
,learning_ratemVmWmXmYmZm[m\m]v^v_v`vavbvcvdve"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
J
s0
t1
u2
v3
w4
x5"
trackable_list_wrapper
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
	regularization_losses
f__call__
h_default_save_signature
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
,
yserving_default"
signature_map
": [
2dense_572/kernel
:
2dense_572/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
i__call__
zactivity_regularizer_fn
*j&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
": 

2dense_573/kernel
:
2dense_573/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
k__call__
|activity_regularizer_fn
*l&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
": 

2dense_574/kernel
:
2dense_574/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
m__call__
~activity_regularizer_fn
*n&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
": 

2dense_575/kernel
:
2dense_575/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
 	variables
!trainable_variables
"regularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
$	variables
%trainable_variables
&regularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Mtotal
	Ncount
O	variables
P	keras_api"
_tf_keras_metric
^
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
U	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
M0
N1"
trackable_list_wrapper
-
O	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
-
T	variables"
_generic_user_object
':%[
2Adam/dense_572/kernel/m
!:
2Adam/dense_572/bias/m
':%

2Adam/dense_573/kernel/m
!:
2Adam/dense_573/bias/m
':%

2Adam/dense_574/kernel/m
!:
2Adam/dense_574/bias/m
':%

2Adam/dense_575/kernel/m
!:
2Adam/dense_575/bias/m
':%[
2Adam/dense_572/kernel/v
!:
2Adam/dense_572/bias/v
':%

2Adam/dense_573/kernel/v
!:
2Adam/dense_573/bias/v
':%

2Adam/dense_574/kernel/v
!:
2Adam/dense_574/bias/v
':%

2Adam/dense_575/kernel/v
!:
2Adam/dense_575/bias/v
?2?
1__inference_sequential_143_layer_call_fn_44352323
1__inference_sequential_143_layer_call_fn_44352833
1__inference_sequential_143_layer_call_fn_44352857
1__inference_sequential_143_layer_call_fn_44352566?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352965
L__inference_sequential_143_layer_call_and_return_conditional_losses_44353073
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352655
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352744?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference__wrapped_model_44352080dense_572_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_572_layer_call_fn_44353094?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_dense_572_layer_call_and_return_all_conditional_losses_44353105?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_573_layer_call_fn_44353126?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_dense_573_layer_call_and_return_all_conditional_losses_44353137?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_574_layer_call_fn_44353158?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_dense_574_layer_call_and_return_all_conditional_losses_44353169?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_575_layer_call_fn_44353178?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_575_layer_call_and_return_conditional_losses_44353188?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_softmax_143_layer_call_fn_44353193?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_softmax_143_layer_call_and_return_conditional_losses_44353198?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_44353209?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_44353220?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_44353231?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_44353242?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_44353253?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_44353264?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
&__inference_signature_wrapper_44352809dense_572_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_dense_572_activity_regularizer_44352093?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
G__inference_dense_572_layer_call_and_return_conditional_losses_44353287?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_dense_573_activity_regularizer_44352106?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
G__inference_dense_573_layer_call_and_return_conditional_losses_44353310?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_dense_574_activity_regularizer_44352119?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
G__inference_dense_574_layer_call_and_return_conditional_losses_44353333?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_443520808?5
.?+
)?&
dense_572_input?????????[
? "9?6
4
softmax_143%?"
softmax_143?????????
]
3__inference_dense_572_activity_regularizer_44352093&?
?
?	
x
? "? ?
K__inference_dense_572_layer_call_and_return_all_conditional_losses_44353105j/?,
%?"
 ?
inputs?????????[
? "3?0
?
0?????????

?
?	
1/0 ?
G__inference_dense_572_layer_call_and_return_conditional_losses_44353287\/?,
%?"
 ?
inputs?????????[
? "%?"
?
0?????????

? 
,__inference_dense_572_layer_call_fn_44353094O/?,
%?"
 ?
inputs?????????[
? "??????????
]
3__inference_dense_573_activity_regularizer_44352106&?
?
?	
x
? "? ?
K__inference_dense_573_layer_call_and_return_all_conditional_losses_44353137j/?,
%?"
 ?
inputs?????????

? "3?0
?
0?????????

?
?	
1/0 ?
G__inference_dense_573_layer_call_and_return_conditional_losses_44353310\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????

? 
,__inference_dense_573_layer_call_fn_44353126O/?,
%?"
 ?
inputs?????????

? "??????????
]
3__inference_dense_574_activity_regularizer_44352119&?
?
?	
x
? "? ?
K__inference_dense_574_layer_call_and_return_all_conditional_losses_44353169j/?,
%?"
 ?
inputs?????????

? "3?0
?
0?????????

?
?	
1/0 ?
G__inference_dense_574_layer_call_and_return_conditional_losses_44353333\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????

? 
,__inference_dense_574_layer_call_fn_44353158O/?,
%?"
 ?
inputs?????????

? "??????????
?
G__inference_dense_575_layer_call_and_return_conditional_losses_44353188\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????

? 
,__inference_dense_575_layer_call_fn_44353178O/?,
%?"
 ?
inputs?????????

? "??????????
=
__inference_loss_fn_0_44353209?

? 
? "? =
__inference_loss_fn_1_44353220?

? 
? "? =
__inference_loss_fn_2_44353231?

? 
? "? =
__inference_loss_fn_3_44353242?

? 
? "? =
__inference_loss_fn_4_44353253?

? 
? "? =
__inference_loss_fn_5_44353264?

? 
? "? ?
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352655?@?=
6?3
)?&
dense_572_input?????????[
p 

 
? "O?L
?
0?????????

-?*
?	
1/0 
?	
1/1 
?	
1/2 ?
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352744?@?=
6?3
)?&
dense_572_input?????????[
p

 
? "O?L
?
0?????????

-?*
?	
1/0 
?	
1/1 
?	
1/2 ?
L__inference_sequential_143_layer_call_and_return_conditional_losses_44352965?7?4
-?*
 ?
inputs?????????[
p 

 
? "O?L
?
0?????????

-?*
?	
1/0 
?	
1/1 
?	
1/2 ?
L__inference_sequential_143_layer_call_and_return_conditional_losses_44353073?7?4
-?*
 ?
inputs?????????[
p

 
? "O?L
?
0?????????

-?*
?	
1/0 
?	
1/1 
?	
1/2 ?
1__inference_sequential_143_layer_call_fn_44352323f@?=
6?3
)?&
dense_572_input?????????[
p 

 
? "??????????
?
1__inference_sequential_143_layer_call_fn_44352566f@?=
6?3
)?&
dense_572_input?????????[
p

 
? "??????????
?
1__inference_sequential_143_layer_call_fn_44352833]7?4
-?*
 ?
inputs?????????[
p 

 
? "??????????
?
1__inference_sequential_143_layer_call_fn_44352857]7?4
-?*
 ?
inputs?????????[
p

 
? "??????????
?
&__inference_signature_wrapper_44352809?K?H
? 
A?>
<
dense_572_input)?&
dense_572_input?????????["9?6
4
softmax_143%?"
softmax_143?????????
?
I__inference_softmax_143_layer_call_and_return_conditional_losses_44353198\3?0
)?&
 ?
inputs?????????


 
? "%?"
?
0?????????

? ?
.__inference_softmax_143_layer_call_fn_44353193O3?0
)?&
 ?
inputs?????????


 
? "??????????
