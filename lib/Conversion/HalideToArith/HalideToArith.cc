#include "mlir/Conversion/HalideToArith/HalideToArith.hh"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Halide/IR/HalideOps.hh"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_DEF_CONVERTHALIDETOARITH
#include "mlir/Conversion/Conversions.h.inc"

using namespace mlir;
using namespace mlir::halide;

namespace {

//===----------------------------------------------------------------------===//
// Binary Arithmetic Conversions
//===----------------------------------------------------------------------===//

template <typename HalideOp, typename IntOp, typename FloatOp>
struct BinaryArithOpConversion : OpConversionPattern<HalideOp> {
    using OpConversionPattern<HalideOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(HalideOp op,
                    typename BinaryArithOpConversion::OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto resultType = op.getType();

        if (isa<IntegerType>(resultType)) {
            rewriter.replaceOpWithNewOp<IntOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs());
        } else if (isa<FloatType>(resultType)) {
            rewriter.replaceOpWithNewOp<FloatOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
        } else {
            return failure();
        }

        return success();
    }
};

// Arithmetic operations
using AddOpConversion =
    BinaryArithOpConversion<AddOp, arith::AddIOp, arith::AddFOp>;
using SubOpConversion =
    BinaryArithOpConversion<SubOp, arith::SubIOp, arith::SubFOp>;
using MulOpConversion =
    BinaryArithOpConversion<MulOp, arith::MulIOp, arith::MulFOp>;
using DivOpConversion =
    BinaryArithOpConversion<DivOp, arith::DivSIOp, arith::DivFOp>;
using ModOpConversion =
    BinaryArithOpConversion<ModOp, arith::RemSIOp, arith::RemFOp>;

// Min/Max operations
template <typename HalideOp, typename IntOp, typename FloatOp>
struct MinMaxOpConversion : OpConversionPattern<HalideOp> {
    using OpConversionPattern<HalideOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(HalideOp op, typename MinMaxOpConversion::OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto resultType = op.getType();

        if (isa<IntegerType>(resultType)) {
            rewriter.replaceOpWithNewOp<IntOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs());
        } else if (isa<FloatType>(resultType)) {
            rewriter.replaceOpWithNewOp<FloatOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
        } else {
            return failure();
        }

        return success();
    }
};

using MinOpConversion =
    MinMaxOpConversion<MinOp, arith::MinSIOp, arith::MinimumFOp>;
using MaxOpConversion =
    MinMaxOpConversion<MaxOp, arith::MaxSIOp, arith::MaximumFOp>;

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

template <typename HalideOp, arith::CmpIPredicate IntPred,
          arith::CmpFPredicate FloatPred>
struct CompareOpConversion : OpConversionPattern<HalideOp> {
    using OpConversionPattern<HalideOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(HalideOp op,
                    typename CompareOpConversion::OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto lhsType = adaptor.getLhs().getType();

        if (isa<IntegerType>(lhsType)) {
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(
                op, IntPred, adaptor.getLhs(), adaptor.getRhs());
        } else if (isa<FloatType>(lhsType)) {
            rewriter.replaceOpWithNewOp<arith::CmpFOp>(
                op, FloatPred, adaptor.getLhs(), adaptor.getRhs());
        } else {
            return failure();
        }

        return success();
    }
};

using EQOpConversion = CompareOpConversion<EQOp, arith::CmpIPredicate::eq,
                                           arith::CmpFPredicate::OEQ>;
using NEOpConversion = CompareOpConversion<NEOp, arith::CmpIPredicate::ne,
                                           arith::CmpFPredicate::ONE>;
using LTOpConversion = CompareOpConversion<LTOp, arith::CmpIPredicate::slt,
                                           arith::CmpFPredicate::OLT>;
using LEOpConversion = CompareOpConversion<LEOp, arith::CmpIPredicate::sle,
                                           arith::CmpFPredicate::OLE>;
using GTOpConversion = CompareOpConversion<GTOp, arith::CmpIPredicate::sgt,
                                           arith::CmpFPredicate::OGT>;
using GEOpConversion = CompareOpConversion<GEOp, arith::CmpIPredicate::sge,
                                           arith::CmpFPredicate::OGE>;

//===----------------------------------------------------------------------===//
// Logical Operations
//===----------------------------------------------------------------------===//

struct AndOpConversion : OpConversionPattern<AndOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(AndOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::AndIOp>(op, adaptor.getLhs(),
                                                   adaptor.getRhs());
        return success();
    }
};

struct OrOpConversion : OpConversionPattern<OrOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(OrOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::OrIOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs());
        return success();
    }
};

struct NotOpConversion : OpConversionPattern<NotOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(NotOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        // Create a constant with all bits set
        auto type = adaptor.getValue().getType();
        Value allOnes;

        if (isa<IntegerType>(type)) {
            allOnes = rewriter.create<arith::ConstantOp>(
                op.getLoc(), rewriter.getIntegerAttr(type, -1));
        } else {
            return failure();
        }

        rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, adaptor.getValue(),
                                                   allOnes);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Select Operation
//===----------------------------------------------------------------------===//

struct SelectOpConversion : OpConversionPattern<SelectOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(SelectOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::SelectOp>(op, adaptor.getCondition(),
                                                     adaptor.getTrueValue(),
                                                     adaptor.getFalseValue());
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Cast Operation
//===----------------------------------------------------------------------===//

struct CastOpConversion : OpConversionPattern<CastOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CastOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto srcType = adaptor.getValue().getType();
        auto dstType = op.getType();

        auto srcInt = dyn_cast<IntegerType>(srcType);
        auto dstInt = dyn_cast<IntegerType>(dstType);
        auto srcFloat = dyn_cast<FloatType>(srcType);
        auto dstFloat = dyn_cast<FloatType>(dstType);

        // Integer to Integer
        if (srcInt && dstInt) {
            if (srcInt.getWidth() < dstInt.getWidth()) {
                // Sign extension
                rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, dstType,
                                                            adaptor.getValue());
            } else if (srcInt.getWidth() > dstInt.getWidth()) {
                // Truncation
                rewriter.replaceOpWithNewOp<arith::TruncIOp>(
                    op, dstType, adaptor.getValue());
            } else {
                // Same width, no-op
                rewriter.replaceOp(op, adaptor.getValue());
            }
            return success();
        }

        // Float to Float
        if (srcFloat && dstFloat) {
            if (srcFloat.getWidth() < dstFloat.getWidth()) {
                rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, dstType,
                                                           adaptor.getValue());
            } else if (srcFloat.getWidth() > dstFloat.getWidth()) {
                rewriter.replaceOpWithNewOp<arith::TruncFOp>(
                    op, dstType, adaptor.getValue());
            } else {
                rewriter.replaceOp(op, adaptor.getValue());
            }
            return success();
        }

        // Integer to Float
        if (srcInt && dstFloat) {
            rewriter.replaceOpWithNewOp<arith::SIToFPOp>(op, dstType,
                                                         adaptor.getValue());
            return success();
        }

        // Float to Integer
        if (srcFloat && dstInt) {
            rewriter.replaceOpWithNewOp<arith::FPToSIOp>(op, dstType,
                                                         adaptor.getValue());
            return success();
        }

        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ConvertHalideToArithPass
    : ::impl::ConvertHalideToArithBase<ConvertHalideToArithPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        populateHalideToArithConversionPatterns(patterns);

        ConversionTarget target(getContext());
        target.addLegalDialect<arith::ArithDialect>();
        target.addIllegalOp<AddOp, SubOp, MulOp, DivOp, ModOp>();
        target.addIllegalOp<MinOp, MaxOp>();
        target.addIllegalOp<EQOp, NEOp, LTOp, LEOp, GTOp, GEOp>();
        target.addIllegalOp<AndOp, OrOp, NotOp>();
        target.addIllegalOp<SelectOp, CastOp>();

        if (failed(applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertHalideToArith() {
    return std::make_unique<ConvertHalideToArithPass>();
}
} // namespace mlir

namespace mlir::halide {
void populateHalideToArithConversionPatterns(RewritePatternSet &patterns) {
    // Arithmetic operations
    patterns.add<AddOpConversion, SubOpConversion, MulOpConversion,
                 DivOpConversion, ModOpConversion>(patterns.getContext());

    // Min/Max operations
    patterns.add<MinOpConversion, MaxOpConversion>(patterns.getContext());

    // Comparison operations
    patterns.add<EQOpConversion, NEOpConversion, LTOpConversion, LEOpConversion,
                 GTOpConversion, GEOpConversion>(patterns.getContext());

    // Logical operations
    patterns.add<AndOpConversion, OrOpConversion, NotOpConversion>(
        patterns.getContext());

    // Select and Cast
    patterns.add<SelectOpConversion, CastOpConversion>(patterns.getContext());
}
} // namespace mlir::halide
