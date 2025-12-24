include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Halide Dialect Definition
//===----------------------------------------------------------------------===//

def Halide_Dialect : Dialect {
  let name = "halide";
  let summary = "A dialect for Halide IR.";
  let description = [{
    This dialect mirrors the Halide Internal IR as defined in `src/IR.h`.
  }];
  let cppNamespace = "::mlir::halide";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

def Halide_ForType : I32EnumAttr<"ForType", "Types of Halide loops",
  [
    I32EnumAttrCase<"Serial", 0>,
    I32EnumAttrCase<"Parallel", 1>,
    I32EnumAttrCase<"Vectorized", 2>,
    I32EnumAttrCase<"Unrolled", 3>,
    I32EnumAttrCase<"Extern", 4>,
    I32EnumAttrCase<"GPUBlock", 5>,
    I32EnumAttrCase<"GPUThread", 6>,
    I32EnumAttrCase<"GPULane", 7>
  ]> {
  let cppNamespace = "::mlir::halide";
}

def Halide_DeviceAPI : I32EnumAttr<"DeviceAPI", "Target device API",
  [
    I32EnumAttrCase<"None", 0>,
    I32EnumAttrCase<"Host", 1>,
    I32EnumAttrCase<"Default_GPU", 2>,
    I32EnumAttrCase<"CUDA", 3>,
    I32EnumAttrCase<"OpenCL", 4>,
    I32EnumAttrCase<"Metal", 5>,
    I32EnumAttrCase<"Hexagon", 6>,
    I32EnumAttrCase<"HexagonDma", 7>,
    I32EnumAttrCase<"D3D12Compute", 8>,
    I32EnumAttrCase<"Vulkan", 9>,
    I32EnumAttrCase<"WebGPU", 10>
  ]> {
  let cppNamespace = "::mlir::halide";
}

def Halide_MemoryType : I32EnumAttr<"MemoryType", "Memory location type",
  [
    I32EnumAttrCase<"Auto", 0>,
    I32EnumAttrCase<"Heap", 1>,
    I32EnumAttrCase<"Stack", 2>,
    I32EnumAttrCase<"Register", 3>,
    I32EnumAttrCase<"GPUShared", 4>,
    I32EnumAttrCase<"GPUTexture", 5>,
    I32EnumAttrCase<"LockedCache", 6>,
    I32EnumAttrCase<"VTCM", 7>,
    I32EnumAttrCase<"AMXTile", 8>
  ]> {
  let cppNamespace = "::mlir::halide";
}

def Halide_Partition : I32EnumAttr<"Partition", "Loop partitioning policy",
  [
    I32EnumAttrCase<"Auto", 0>,
    I32EnumAttrCase<"Never", 1>,
    I32EnumAttrCase<"Always", 2>
  ]> {
  let cppNamespace = "::mlir::halide";
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

class Halide_Op<string mnemonic, list<Trait> traits = []> :
    Op<Halide_Dialect, mnemonic, traits>;

// Base class for binary arithmetic operations.
// Enforces that lhs, rhs, and result have the same type.
class Halide_BinaryArithmeticOp<string mnemonic, list<Trait> traits = []> :
    Halide_Op<mnemonic, !listconcat(traits, [Pure, SameOperandsAndResultType])> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// Base class for comparison operations.
// Enforces that inputs have the same type, but result is boolean (I1).
class Halide_CompareOp<string mnemonic, list<Trait> traits = []> :
    Halide_Op<mnemonic, !listconcat(traits, [Pure, SameTypeOperands])> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

//===----------------------------------------------------------------------===//
// Control Flow & Terminator
//===----------------------------------------------------------------------===//

def Halide_YieldOp : Halide_Op<"yield", [Terminator]> {
    let summary = "Terminates a Halide region.";
    let description = [{
        The `yield` operation terminates regions in Halide control flow operations
        such as `halide.for` and `halide.if`.
    }];
    let builders = [
      OpBuilder<(ins), [{ /* body is empty, implicitly empty operands */ }]>
    ];
    let arguments = (ins Variadic<AnyType>:$results);
    let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

//===----------------------------------------------------------------------===//
// Expression Operations
//===----------------------------------------------------------------------===//

def Halide_VariableOp : Halide_Op<"variable", [Pure]> {
    let summary = "A named variable.";
    let arguments = (ins StrAttr:$name);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$name attr-dict `:` type($result)";
}

def Halide_LoadOp : Halide_Op<"load"> {
    let summary = "Load a value from a named buffer.";
    let arguments = (ins StrAttr:$name, AnyType:$index, I1:$predicate);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$name `[` $index `:` type($index) `]` `if` $predicate attr-dict `:` type($result)";
}

def Halide_CastOp : Halide_Op<"cast", [Pure]> {
    let summary = "Cast a node from one type to another.";
    let arguments = (ins AnyType:$value);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$value attr-dict `:` type($value) `to` type($result)";
}

def Halide_NotOp : Halide_Op<"not", [Pure, SameOperandsAndResultType]> {
    let summary = "Logical Not";
    let arguments = (ins AnyType:$value);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Halide_SelectOp : Halide_Op<"select", [Pure, AllTypesMatch<["true_value", "false_value", "result"]>]> {
    let summary = "Ternary operator based on condition.";
    let arguments = (ins I1:$condition, AnyType:$true_value, AnyType:$false_value);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$condition `,` $true_value `,` $false_value attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Binary Arithmetic Operations
//===----------------------------------------------------------------------===//

def Halide_AddOp : Halide_BinaryArithmeticOp<"add"> {
    let summary = "The sum of two expressions";
}

def Halide_SubOp : Halide_BinaryArithmeticOp<"sub"> {
    let summary = "The difference of two expressions";
}

def Halide_MulOp : Halide_BinaryArithmeticOp<"mul"> {
    let summary = "The product of two expressions";
}

def Halide_DivOp : Halide_BinaryArithmeticOp<"div"> {
    let summary = "The ratio of two expressions";
}

def Halide_ModOp : Halide_BinaryArithmeticOp<"mod"> {
    let summary = "The remainder of a / b. Always positive.";
}

def Halide_MinOp : Halide_BinaryArithmeticOp<"min"> {
    let summary = "The lesser of two values.";
}

def Halide_MaxOp : Halide_BinaryArithmeticOp<"max"> {
    let summary = "The greater of two values";
}

def Halide_AndOp : Halide_BinaryArithmeticOp<"and"> {
    let summary = "Logical And";
}

def Halide_OrOp : Halide_BinaryArithmeticOp<"or"> {
    let summary = "Logical Or";
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

def Halide_EQOp : Halide_CompareOp<"eq"> {
    let summary = "Equal";
}

def Halide_NEOp : Halide_CompareOp<"ne"> {
    let summary = "Not Equal";
}

def Halide_LTOp : Halide_CompareOp<"lt"> {
    let summary = "Less Than";
}

def Halide_LEOp : Halide_CompareOp<"le"> {
    let summary = "Less Than Or Equal";
}

def Halide_GTOp : Halide_CompareOp<"gt"> {
    let summary = "Greater Than";
}

def Halide_GEOp : Halide_CompareOp<"ge"> {
    let summary = "Greater Than Or Equal";
}

//===----------------------------------------------------------------------===//
// Statement Operations
//===----------------------------------------------------------------------===//

def Halide_IfOp : Halide_Op<"if", [
      SingleBlockImplicitTerminator<"YieldOp">,
      DeclareOpInterfaceMethods<RegionBranchOpInterface>
    ]> {
    let summary = "An if-then-else block.";
    let arguments = (ins I1:$condition);
    let regions = (region SizedRegion<1>:$then_region, AnyRegion:$else_region);

    let assemblyFormat = [{
        $condition $then_region (`else` $else_region^)? attr-dict
    }];
}

def Halide_ForOp : Halide_Op<"for", [
      SingleBlockImplicitTerminator<"YieldOp">,
      DeclareOpInterfaceMethods<LoopLikeOpInterface>
    ]> {
    let summary = "A for loop.";
    let arguments = (ins StrAttr:$name, AnyType:$min, AnyType:$max,
                         Halide_ForType:$for_type, Halide_DeviceAPI:$device_api,
                         Halide_Partition:$partition_policy);

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $name `=` $min `:` type($min) `to` $max `:` type($max)
        `type` $for_type
        `device` $device_api
        $body attr-dict
    }];
}

def Halide_LetStmtOp : Halide_Op<"let_stmt", [SingleBlockImplicitTerminator<"YieldOp">]> {
    let summary = "Defines a new variable for the scope of the body.";
    let arguments = (ins StrAttr:$name, AnyType:$value);
    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $name `=` $value `:` type($value) $body attr-dict
    }];
}

def Halide_StoreOp : Halide_Op<"store"> {
    let summary = "Store a value to a buffer.";
    let arguments = (ins StrAttr:$name, AnyType:$value, AnyType:$index, I1:$predicate);
    let assemblyFormat = [{
        $name `[` $index `:` type($index) `]` `=` $value `if` $predicate attr-dict `:` type($value)
    }];
}

def Halide_AllocateOp : Halide_Op<"allocate", [SingleBlockImplicitTerminator<"YieldOp">]> {
    let summary = "Allocate a scratch area.";
    let arguments = (ins StrAttr:$name, TypeAttr:$type,
                         Halide_MemoryType:$memory_type, Variadic<AnyType>:$extents,
                         I1:$condition);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = [{
        $name `:` $type `[` $extents `:` type($extents) `]` `if` $condition $body attr-dict
    }];
}
