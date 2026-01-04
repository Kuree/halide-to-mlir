#include "mlir/Conversion/HalideToMemRef/HalideToMemRef.hh"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Halide/IR/HalideOps.hh"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringSet.h"

#define GEN_PASS_DEF_CONVERTHALIDETOMEMREF
#include "mlir/Conversion/Conversions.h.inc"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper to identify and convert Halide buffer intrinsic calls
//===----------------------------------------------------------------------===//

bool isBufferHelper(StringRef name) {
    return name.starts_with("_halide_buffer_") || name == "make_struct";
}

/// Base pattern for converting halide. call operations that call buffer helpers
template <typename ConcretePattern>
struct BufferHelperCallConversion : OpConversionPattern<halide::CallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(halide::CallOp callOp, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto funcName = callOp.getName();

        if (!isBufferHelper(funcName)) {
            return failure();
        }

        return ConcretePattern::rewriteBufferHelper(callOp, adaptor, rewriter,
                                                    funcName);
    }
};

//===----------------------------------------------------------------------===//
// _halide_buffer_get_dimensions
// Returns:  int (number of dimensions)
// Conversion: memref.rank
//===----------------------------------------------------------------------===//

struct GetDimensionsConversion
    : BufferHelperCallConversion<GetDimensionsConversion> {
    using BufferHelperCallConversion::BufferHelperCallConversion;

    static LogicalResult
    rewriteBufferHelper(halide::CallOp callOp, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter,
                        StringRef funcName) {
        if (funcName != "_halide_buffer_get_dimensions")
            return failure();

        if (adaptor.getArgs().size() != 1)
            return failure();

        auto loc = callOp.getLoc();
        Value memref = adaptor.getArgs()[0];

        // memref.rank returns index type
        Value rank = rewriter.create<memref::RankOp>(loc, memref);

        rewriter.replaceOpWithNewOp<arith::IndexCastOp>(callOp,
                                                        callOp.getType(), rank);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// _halide_buffer_get_host
// Returns: uint8_t* (pointer to host data)
// Conversion: memref.extract_aligned_pointer_as_index + index_cast
//===----------------------------------------------------------------------===//

struct GetHostConversion : BufferHelperCallConversion<GetHostConversion> {
    using BufferHelperCallConversion::BufferHelperCallConversion;

    static LogicalResult
    rewriteBufferHelper(halide::CallOp callOp, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter,
                        StringRef funcName) {
        if (funcName != "_halide_buffer_get_host")
            return failure();

        if (adaptor.getArgs().size() != 1)
            return failure();

        Value memref = adaptor.getArgs()[0];

        // Extract the base pointer as an index
        rewriter.replaceOpWithNewOp<memref::ExtractAlignedPointerAsIndexOp>(
            callOp, rewriter.getIndexType(), memref);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// _halide_buffer_get_min/max/extent/stride
// Returns: int (dimension property)
// Conversion: memref.dim + constants
//===----------------------------------------------------------------------===//

struct GetDimPropertyConversion
    : BufferHelperCallConversion<GetDimPropertyConversion> {
    using BufferHelperCallConversion::BufferHelperCallConversion;

    static LogicalResult
    rewriteBufferHelper(halide::CallOp callOp, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter,
                        StringRef funcName) {

        auto loc = callOp.getLoc();
        if (adaptor.getArgs().size() != 2)
            return failure();
        auto idxType = rewriter.getIndexType();
        Value memref = adaptor.getArgs()[0];
        Value dimIndex = adaptor.getArgs()[1];
        auto memrefTy = cast<MemRefType>(memref.getType());

        if (funcName == "_halide_buffer_get_min") {
            // return 0? Not sure about subview result
            rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                callOp, callOp.getType(),
                rewriter.getZeroAttr(callOp.getType()));
        } else if (funcName == "_halide_buffer_get_max") {
            // min + extend - 1 since it's inclusive
            auto ty = callOp.getType();
            auto min = rewriter.create<arith::ConstantOp>(
                loc, ty, rewriter.getZeroAttr(ty));
            auto idx = rewriter.createOrFold<arith::IndexCastOp>(loc, idxType,
                                                                 dimIndex);
            auto dim = rewriter.create<memref::DimOp>(loc, memref, idx);
            auto extent = rewriter.createOrFold<arith::IndexCastOp>(
                loc, dimIndex.getType(), dim);
            auto one = rewriter.create<arith::ConstantOp>(
                loc, ty, rewriter.getOneAttr(ty));
            auto add0 = rewriter.createOrFold<arith::AddIOp>(loc, min, extent);
            rewriter.replaceOpWithNewOp<arith::SubIOp>(callOp, add0, one);
        } else if (funcName == "_halide_buffer_get_extent") {
            // memref.dim returns the extent of a dimension
            // same as max dim since we always set min to 0
            auto idx = rewriter.createOrFold<arith::IndexCastOp>(loc, idxType,
                                                                 dimIndex);
            auto dim = rewriter.create<memref::DimOp>(loc, memref, idx);
            rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
                callOp, dimIndex.getType(), dim);
        } else if (funcName == "_halide_buffer_get_stride") {
            // use metadata op. can only handle static value
            auto idxInt = getConstantIntValue(dimIndex);
            if (!idxInt)
                return rewriter.notifyMatchFailure(
                    callOp, "stride dim has to be constant");
            auto rank = memrefTy.getRank();
            auto metadata =
                rewriter.create<memref::ExtractStridedMetadataOp>(loc, memref);
            auto res =
                metadata.getResult(metadata.getNumResults() - rank + *idxInt);
            res =
                rewriter.create<arith::IndexCastOp>(loc, callOp.getType(), res);
            rewriter.replaceOp(callOp, res);
        } else {
            return failure();
        }
        return success();
    }
};

//===----------------------------------------------------------------------===//
// _halide_buffer_get_type
// Returns: uint32_t (encoded type)
// Conversion: Extract from memref element type
//===----------------------------------------------------------------------===//

struct GetTypeConversion : BufferHelperCallConversion<GetTypeConversion> {
    using BufferHelperCallConversion::BufferHelperCallConversion;

    static LogicalResult
    rewriteBufferHelper(halide::CallOp callOp, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter,
                        StringRef funcName) {
        if (funcName != "_halide_buffer_get_type")
            return failure();

        if (adaptor.getArgs().size() != 1)
            return failure();

        auto loc = callOp.getLoc();
        Value memref = adaptor.getArgs()[0];

        // Encode element type as u32 (matching Halide type encoding)
        auto memrefType = dyn_cast<MemRefType>(memref.getType());
        if (!memrefType) {
            return failure();
        }

        Type elementType = memrefType.getElementType();
        uint32_t encodedType = encodeHalideType(elementType);

        Value result = rewriter.create<arith::ConstantIntOp>(loc, encodedType,
                                                             callOp.getType());

        rewriter.replaceOp(callOp, result);
        return success();
    }

  private:
    /// Encode MLIR type to Halide type format
    /// Halide type encoding: bits[0-7]=type_code, bits[8-15]=bits,
    /// bits[16-31]=lanes
    static uint32_t encodeHalideType(Type type) {
        uint32_t code = 0; // Int
        uint32_t bits = 32;
        uint32_t lanes = 1;

        if (auto intType = dyn_cast<IntegerType>(type)) {
            if (intType.isUnsigned()) {
                code = 1; // UInt
            } else {
                code = 0; // Int
            }
            bits = intType.getWidth();
        } else if (auto floatType = dyn_cast<FloatType>(type)) {
            code = 2; // Float
            bits = floatType.getWidth();
        }

        return code & 0xFF | (bits & 0xFF) << 8 | (lanes & 0xFFFF) << 16;
    }
};

//===----------------------------------------------------------------------===//
// _halide_buffer_is_bounds_query
// Returns: bool (true if host==nullptr && device==0)
// Conversion: Check if memref is null
//===----------------------------------------------------------------------===//

struct IsBoundsQueryConversion
    : BufferHelperCallConversion<IsBoundsQueryConversion> {
    using BufferHelperCallConversion::BufferHelperCallConversion;

    static LogicalResult
    rewriteBufferHelper(halide::CallOp callOp, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter,
                        StringRef funcName) {
        if (funcName != "_halide_buffer_is_bounds_query")
            return failure();

        auto loc = callOp.getLoc();
        auto memref = adaptor.getArgs()[0];
        auto idx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
            loc, memref);
        auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

        rewriter.replaceOpWithNewOp<arith::CmpIOp>(
            callOp, arith::CmpIPredicate::eq, idx, zero);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// _halide_buffer_crop
// Creates a subview of a buffer
// Conversion: memref.subview
//===----------------------------------------------------------------------===//

struct BufferCropConversion : BufferHelperCallConversion<BufferCropConversion> {
    using BufferHelperCallConversion::BufferHelperCallConversion;

    static LogicalResult
    rewriteBufferHelper(halide::CallOp callOp, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter,
                        StringRef funcName) {
        if (funcName != "_halide_buffer_crop")
            return failure();

        // _halide_buffer_crop(user_context, dst, dst_shape, src, min, extent)
        if (adaptor.getArgs().size() != 6)
            return failure();

        // we don't support it right now
        return rewriter.notifyMatchFailure(callOp,
                                           "buffer crop not supported for now");
    }
};

//===----------------------------------------------------------------------===//
// _halide_buffer_set_host_dirty / _halide_buffer_set_device_dirty
// _halide_buffer_get_host_dirty / _halide_buffer_get_device_dirty
// These track data coherency between host and device
// Conversion: For CPU-only, these can be no-ops
// For GPU, would need to insert explicit copy operations
//===----------------------------------------------------------------------===//

struct DirtyFlagConversion : BufferHelperCallConversion<DirtyFlagConversion> {
    using BufferHelperCallConversion::BufferHelperCallConversion;

    static LogicalResult
    rewriteBufferHelper(halide::CallOp callOp, OpAdaptor,
                        ConversionPatternRewriter &rewriter,
                        StringRef funcName) {
        if (!funcName.contains("dirty"))
            return failure();

        auto loc = callOp.getLoc();

        if (funcName.contains("set_")) {
            // Setter returns 0 (success)
            Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
            rewriter.replaceOp(callOp, zero);
        } else if (funcName.contains("get_")) {
            // Getter returns false (not dirty)
            Value falseVal = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);
            rewriter.replaceOp(callOp, falseVal);
        }

        return success();
    }
};

//===----------------------------------------------------------------------===//
// _halide_buffer_set_bounds
// Sets min and extent for a dimension
// _halide_buffer_init
// make_struct
// Noop since memref is created by user
//===----------------------------------------------------------------------===//

struct NoopConversion : BufferHelperCallConversion<NoopConversion> {
    using BufferHelperCallConversion::BufferHelperCallConversion;

    static LogicalResult
    rewriteBufferHelper(halide::CallOp callOp, OpAdaptor,
                        ConversionPatternRewriter &rewriter,
                        StringRef funcName) {
        if (funcName != "_halide_buffer_set_bounds" &&
            funcName != "_halide_buffer_init" &&
            funcName != "_halide_buffer_get_shape" && funcName != "make_struct")
            return failure();
        auto ty = callOp.getType();
        if (isa<halide::HandleType>(ty))
            ty = rewriter.getIndexType();
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            callOp, rewriter.getZeroAttr(ty));

        return success();
    }
};

//===----------------------------------------------------------------------===//
// Operation Conversions
//===----------------------------------------------------------------------===//

Value lookUpMemrefValue(StringRef varName, Operation *op) {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    for (auto i = 0; i < funcOp.getNumArguments(); i++) {
        if (auto name = funcOp.getArgAttrOfType<StringAttr>(i, "halide.name")) {
            if (name == varName) {
                return funcOp.getArgument(i);
            }
        }
    }
    return {};
}

memref::CollapseShapeOp collapseShape(MemRefType ty, Value memrefValue,
                                      OpBuilder &builder) {
    ReassociationIndices indices(ty.getRank());
    for (auto i = 0; i < ty.getRank(); i++)
        indices[i] = i;
    SmallVector<ReassociationIndices> reassociation = {indices};
    return builder.create<memref::CollapseShapeOp>(builder.getUnknownLoc(),
                                                   memrefValue, reassociation);
}

struct LoadOpConversion : OpConversionPattern<halide::LoadOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(halide::LoadOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto memrefValue = lookUpMemrefValue(op.getName(), op);
        if (!memrefValue)
            return failure();
        auto idxValue = op.getIndex();
        auto memrefTy = dyn_cast<MemRefType>(memrefValue.getType());
        if (!memrefTy)
            return failure();

        auto ifOp = rewriter.create<scf::IfOp>(
            op.getLoc(), op->getResultTypes(), adaptor.getPredicate(),
            /*addThenBlock=*/true, /*addElseBlock=*/true);

        rewriter.setInsertionPointToEnd(ifOp.thenBlock());

        memrefValue = collapseShape(memrefTy, memrefValue, rewriter);
        idxValue = rewriter.createOrFold<arith::IndexCastOp>(
            rewriter.getUnknownLoc(), rewriter.getIndexType(), idxValue);
        auto loadOp = rewriter.create<memref::LoadOp>(op.getLoc(), op.getType(),
                                                      memrefValue, idxValue);
        rewriter.create<scf::YieldOp>(op.getLoc(), loadOp->getResults());

        rewriter.setInsertionPointToEnd(ifOp.elseBlock());
        // Zero out
        auto zero = rewriter.create<arith::ConstantOp>(
            op.getLoc(), op.getType(), rewriter.getZeroAttr(op.getType()));
        rewriter.create<scf::YieldOp>(op.getLoc(), zero->getResults());

        rewriter.replaceOp(op, ifOp);
        return success();
    }
};

struct StoreOpConversion : OpConversionPattern<halide::StoreOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(halide::StoreOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto memrefValue = lookUpMemrefValue(op.getName(), op);
        if (!memrefValue)
            return failure();

        auto idxValue = op.getIndex();
        auto memrefTy = dyn_cast<MemRefType>(memrefValue.getType());
        if (!memrefTy)
            return failure();

        auto ifOp = rewriter.create<scf::IfOp>(
            op.getLoc(), op->getResultTypes(), adaptor.getPredicate(),
            /*addThenBlock=*/true, /*addElseBlock=*/false);

        rewriter.setInsertionPointToEnd(ifOp.thenBlock());

        memrefValue = collapseShape(memrefTy, memrefValue, rewriter);
        idxValue = rewriter.createOrFold<arith::IndexCastOp>(
            rewriter.getUnknownLoc(), rewriter.getIndexType(), idxValue);
        rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.getValue(),
                                         memrefValue, idxValue);
        rewriter.create<scf::YieldOp>(op.getLoc());

        rewriter.replaceOp(op, ifOp);

        return success();
    }
};

struct CastOpConversion : OpConversionPattern<halide::CastOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(halide::CastOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        if (isa<MemRefType>(adaptor.getValue().getType()) &&
            op.getType().isInteger()) {
            rewriter.replaceOpWithNewOp<memref::ExtractAlignedPointerAsIndexOp>(
                op, adaptor.getValue());
            return success();
        }
        rewriter.replaceOp(op, adaptor.getValue());
        return success();
    }
};

struct ConvertHalideToMemRefPass
    : ::impl::ConvertHalideToMemRefBase<ConvertHalideToMemRefPass> {

    void runOnOperation() override {
        TypeConverter typeConverter;
        RewritePatternSet patterns(&getContext());
        ConversionTarget target(getContext());

        // Mark MemRef and Arith dialects as legal
        target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect,
                               func::FuncDialect, scf::SCFDialect>();

        // Mark Halide load/store/call as illegal
        target.addIllegalOp<halide::LoadOp, halide::StoreOp>();

        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            // Function is legal if it doesn't use halide.buffer types
            return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                   typeConverter.isLegal(&op.getBody());
        });

        target.addDynamicallyLegalOp<halide::CastOp>([&](halide::CastOp op) {
            return op.getValue().getType().isIntOrFloat();
        });

        target.addDynamicallyLegalOp<halide::CallOp>([](halide::CallOp op) {
            StringRef name = op.getName();
            return !isBufferHelper(name);
        });

        // Populate conversion patterns
        halide::populateHalideToMemRefConversionPatterns(typeConverter,
                                                         patterns);
        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
            patterns, typeConverter);

        if (failed(applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertHalideToMemRef() {
    return std::make_unique<ConvertHalideToMemRefPass>();
}
} // namespace mlir

namespace mlir::halide {
void populateHalideToMemRefConversionPatterns(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns) {
    // Keep all other types unchanged
    typeConverter.addConversion([](Type type) { return type; });

    // Convert halide.buffer to memref
    typeConverter.addConversion([](BufferType type) {
        int64_t rank = type.getRank();
        Type elementType = type.getElementType();

        // Create a shape with all dynamic dimensions
        SmallVector<int64_t> shape(rank, ShapedType::kDynamic);

        return MemRefType::get(shape, elementType);
    });

    typeConverter.addConversion(
        [](HandleType type) { return IndexType::get(type.getContext()); });

    typeConverter.addSourceMaterialization(
        [](OpBuilder &builder, Type ty, ValueRange values,
           Location loc) -> std::optional<Value> {
            if (values.size() != 1)
                return {};
            auto v = values.front();
            if (isa<IntegerType>(ty) && isa<IndexType>(v.getType())) {
                return builder.create<arith::IndexCastOp>(loc, ty, v);
            }
            return {};
        });

    patterns.add<GetDimensionsConversion, GetHostConversion,
                 GetDimPropertyConversion, GetTypeConversion,
                 IsBoundsQueryConversion, BufferCropConversion,
                 DirtyFlagConversion, NoopConversion, LoadOpConversion,
                 StoreOpConversion, CastOpConversion>(typeConverter,
                                                      patterns.getContext());
}

} // namespace mlir::halide
