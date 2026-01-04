#include "mlir/Conversion/HalideToMath/HalideToMath.hh"

#include "mlir/Dialect/Halide/IR/HalideOps.hh"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringSet.h"

#define GEN_PASS_DEF_CONVERTHALIDETOMATH
#include "mlir/Conversion/Conversions.h.inc"

using namespace mlir;
using namespace mlir::halide;

namespace {
//===----------------------------------------------------------------------===//
// CallOp to Math/LLVM Dialect Conversions
//===----------------------------------------------------------------------===//

// Helper to check if a CallOp is a pure intrinsic function
bool isPureIntrinsic(CallOp op) {
    return op.getCallType() == CallType::PureIntrinsic ||
           op.getCallType() == CallType::Intrinsic;
}

struct UnaryCallOpToArithConversion : OpConversionPattern<CallOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(CallOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        if (!isPureIntrinsic(op)) {
            return failure();
        }

        StringRef name = op.getName();
        auto resultType = op.getType();
        auto args = adaptor.getArgs();
        if (args.size() != 1)
            return failure();
        Value arg = args[0];

        // abs - works for both int and float
        if (name == "abs") {
            if (isa<FloatType>(resultType)) {
                rewriter.replaceOpWithNewOp<math::AbsFOp>(op, resultType, arg);
            } else if (isa<IntegerType>(resultType)) {
                rewriter.replaceOpWithNewOp<math::AbsIOp>(op, resultType, arg);
            } else {
                return failure();
            }
            return success();
        }

        // Integer bit operations
        if (isa<IntegerType>(resultType)) {
            if (name == "count_leading_zeros") {
                rewriter.replaceOpWithNewOp<math::CountLeadingZerosOp>(
                    op, resultType, arg);
                return success();
            }
            if (name == "count_trailing_zeros") {
                rewriter.replaceOpWithNewOp<math::CountTrailingZerosOp>(
                    op, resultType, arg);
                return success();
            }
        }

        // Floating point operations
        if (isa<FloatType>(resultType)) {
            if (name == "round") {
                rewriter.replaceOpWithNewOp<math::RoundEvenOp>(op, resultType,
                                                               arg);
                return success();
            }
        }
        return failure();
    }
};

struct BinaryCallOpToArithConversion : OpConversionPattern<CallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CallOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        if (!isPureIntrinsic(op)) {
            return failure();
        }

        StringRef name = op.getName();
        auto resultType = op.getType();
        auto args = adaptor.getArgs();
        if (args.size() != 2)
            return failure();
        Value lhs = args[0];
        Value rhs = args[1];

        // absd - absolute difference
        if (name == "absd") {
            if (isa<IntegerType>(resultType)) {
                auto sub =
                    rewriter.create<arith::SubIOp>(op.getLoc(), lhs, rhs);
                rewriter.replaceOpWithNewOp<math::AbsIOp>(op, resultType, sub);
                return success();
            }
            if (isa<FloatType>(resultType)) {
                auto sub =
                    rewriter.create<arith::SubFOp>(op.getLoc(), lhs, rhs);
                rewriter.replaceOpWithNewOp<math::AbsFOp>(op, resultType, sub);
                return success();
            }
        }
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ConvertHalideToMathPass
    : ::impl::ConvertHalideToMathBase<ConvertHalideToMathPass> {
    using ConvertHalideToMathBase::ConvertHalideToMathBase;

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns
            .add<UnaryCallOpToArithConversion, BinaryCallOpToArithConversion>(
                &getContext());

        ConversionTarget target(getContext());
        target.addLegalDialect<math::MathDialect, arith::ArithDialect>();
        target.addDynamicallyLegalOp<CallOp>([](CallOp op) {
            if (op.getCallType() != CallType::PureIntrinsic &&
                op.getCallType() != CallType::Intrinsic) {
                return true;
            }

            StringRef name = op.getName();
            static const llvm::StringSet mathIntrinsics = {
                "abs", "absd", "round", "count_leading_zeros",
                "count_trailing_zeros"};

            return !mathIntrinsics.contains(name);
        });

        if (failed(applyPartialConversion(getOperation(), target,
                                          std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertHalideToMath() {
    return std::make_unique<ConvertHalideToMathPass>();
}
} // namespace mlir

namespace mlir::halide {
void populateHalideCallToMathConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<UnaryCallOpToArithConversion, BinaryCallOpToArithConversion>(
        patterns.getContext());
}
} // namespace mlir::halide
