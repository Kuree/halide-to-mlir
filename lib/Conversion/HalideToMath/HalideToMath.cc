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
// CallOp to Math Dialect Conversions
//===----------------------------------------------------------------------===//

// Helper to check if a CallOp is a pure intrinsic math function
bool isPureMathIntrinsic(CallOp op) {
    return op.getCallType() == CallType::PureIntrinsic ||
           op.getCallType() == CallType::Intrinsic;
}

// Base pattern for converting Halide CallOp to Math dialect ops
template <typename MathOp>
struct CallOpToMathPattern : OpConversionPattern<CallOp> {
    using OpConversionPattern::OpConversionPattern;

    StringRef intrinsicName;
    size_t numArgs;

    CallOpToMathPattern(MLIRContext *ctx, StringRef name, size_t args)
        : OpConversionPattern(ctx), intrinsicName(name), numArgs(args) {}

    LogicalResult
    matchAndRewrite(CallOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        // Check if this is the right intrinsic
        if (op.getName() != intrinsicName) {
            return failure();
        }

        // Check if it's a pure intrinsic
        if (!isPureMathIntrinsic(op)) {
            return failure();
        }

        // Check argument count
        if (adaptor.getArgs().size() != numArgs) {
            return failure();
        }

        // Only handle float types for now
        if (!isa<FloatType>(op.getType())) {
            return failure();
        }

        // Create the math dialect operation
        rewriter.replaceOpWithNewOp<MathOp>(op, op.getType(),
                                            adaptor.getArgs());
        return success();
    }
};

// Specialized pattern for unary math operations
struct UnaryMathOpConversion : OpConversionPattern<CallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CallOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        if (!isPureMathIntrinsic(op)) {
            return failure();
        }

        if (op.getArgs().size() != 1) {
            return failure();
        }

        auto resultType = op.getType();
        if (!isa<FloatType>(resultType)) {
            return failure();
        }

        StringRef name = op.getName();
        Value arg = adaptor.getArgs()[0];

        // Map Halide intrinsic names to Math dialect ops
        if (name == "sqrt_f32" || name == "sqrt_f64" || name == "sqrt") {
            rewriter.replaceOpWithNewOp<math::SqrtOp>(op, resultType, arg);
        } else if (name == "sin_f32" || name == "sin_f64" || name == "sin") {
            rewriter.replaceOpWithNewOp<math::SinOp>(op, resultType, arg);
        } else if (name == "cos_f32" || name == "cos_f64" || name == "cos") {
            rewriter.replaceOpWithNewOp<math::CosOp>(op, resultType, arg);
        } else if (name == "tan_f32" || name == "tan_f64" || name == "tan") {
            rewriter.replaceOpWithNewOp<math::TanOp>(op, resultType, arg);
        } else if (name == "exp_f32" || name == "exp_f64" || name == "exp") {
            rewriter.replaceOpWithNewOp<math::ExpOp>(op, resultType, arg);
        } else if (name == "log_f32" || name == "log_f64" || name == "log") {
            rewriter.replaceOpWithNewOp<math::LogOp>(op, resultType, arg);
        } else if (name == "log2_f32" || name == "log2_f64" || name == "log2") {
            rewriter.replaceOpWithNewOp<math::Log2Op>(op, resultType, arg);
        } else if (name == "log10_f32" || name == "log10_f64" ||
                   name == "log10") {
            rewriter.replaceOpWithNewOp<math::Log10Op>(op, resultType, arg);
        } else if (name == "abs_f32" || name == "abs_f64" || name == "fabs") {
            rewriter.replaceOpWithNewOp<math::AbsFOp>(op, resultType, arg);
        } else if (name == "floor_f32" || name == "floor_f64" ||
                   name == "floor") {
            rewriter.replaceOpWithNewOp<math::FloorOp>(op, resultType, arg);
        } else if (name == "ceil_f32" || name == "ceil_f64" || name == "ceil") {
            rewriter.replaceOpWithNewOp<math::CeilOp>(op, resultType, arg);
        } else if (name == "round_f32" || name == "round_f64" ||
                   name == "round") {
            rewriter.replaceOpWithNewOp<math::RoundOp>(op, resultType, arg);
        } else if (name == "trunc_f32" || name == "trunc_f64" ||
                   name == "trunc") {
            rewriter.replaceOpWithNewOp<math::TruncOp>(op, resultType, arg);
        } else if (name == "asin_f32" || name == "asin_f64" || name == "asin") {
            rewriter.replaceOpWithNewOp<math::AsinOp>(op, resultType, arg);
        } else if (name == "acos_f32" || name == "acos_f64" || name == "acos") {
            rewriter.replaceOpWithNewOp<math::AcosOp>(op, resultType, arg);
        } else if (name == "atan_f32" || name == "atan_f64" || name == "atan") {
            rewriter.replaceOpWithNewOp<math::AtanOp>(op, resultType, arg);
        } else if (name == "sinh_f32" || name == "sinh_f64" || name == "sinh") {
            rewriter.replaceOpWithNewOp<math::SinhOp>(op, resultType, arg);
        } else if (name == "cosh_f32" || name == "cosh_f64" || name == "cosh") {
            rewriter.replaceOpWithNewOp<math::CoshOp>(op, resultType, arg);
        } else if (name == "tanh_f32" || name == "tanh_f64" || name == "tanh") {
            rewriter.replaceOpWithNewOp<math::TanhOp>(op, resultType, arg);
        } else if (name == "erf_f32" || name == "erf_f64" || name == "erf") {
            rewriter.replaceOpWithNewOp<math::ErfOp>(op, resultType, arg);
        } else if (name == "exp2_f32" || name == "exp2_f64" || name == "exp2") {
            rewriter.replaceOpWithNewOp<math::Exp2Op>(op, resultType, arg);
        } else if (name == "rsqrt_f32" || name == "rsqrt_f64" ||
                   name == "fast_inverse_sqrt") {
            rewriter.replaceOpWithNewOp<math::RsqrtOp>(op, resultType, arg);
        } else {
            return failure();
        }

        return success();
    }
};

// Specialized pattern for binary math operations
struct BinaryMathOpConversion : OpConversionPattern<CallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CallOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        if (!isPureMathIntrinsic(op)) {
            return failure();
        }

        if (op.getArgs().size() != 2) {
            return failure();
        }

        auto resultType = op.getType();
        if (!isa<FloatType>(resultType)) {
            return failure();
        }

        StringRef name = op.getName();
        Value lhs = adaptor.getArgs()[0];
        Value rhs = adaptor.getArgs()[1];

        // Map Halide intrinsic names to Math dialect ops
        if (name == "pow_f32" || name == "pow_f64" || name == "pow") {
            rewriter.replaceOpWithNewOp<math::PowFOp>(op, resultType, lhs, rhs);
        } else if (name == "atan2_f32" || name == "atan2_f64" ||
                   name == "atan2") {
            rewriter.replaceOpWithNewOp<math::Atan2Op>(op, resultType, lhs,
                                                       rhs);
        } else if (name == "copysign_f32" || name == "copysign_f64" ||
                   name == "copysign") {
            rewriter.replaceOpWithNewOp<math::CopySignOp>(op, resultType, lhs,
                                                          rhs);
        } else {
            return failure();
        }

        return success();
    }
};

// Specialized pattern for ternary math operations (e.g., fma)
struct TernaryMathOpConversion : OpConversionPattern<CallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CallOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        if (!isPureMathIntrinsic(op)) {
            return failure();
        }

        if (op.getArgs().size() != 3) {
            return failure();
        }

        auto resultType = op.getType();
        if (!isa<FloatType>(resultType)) {
            return failure();
        }

        StringRef name = op.getName();

        // Map Halide intrinsic names to Math dialect ops
        if (name == "fma_f32" || name == "fma_f64" || name == "fma") {
            rewriter.replaceOpWithNewOp<math::FmaOp>(
                op, resultType, adaptor.getArgs()[0], adaptor.getArgs()[1],
                adaptor.getArgs()[2]);
        } else {
            return failure();
        }

        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ConvertHalideToMathPass
    : ::impl::ConvertHalideToMathBase<ConvertHalideToMathPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        populateHalideCallToMathConversionPatterns(patterns);

        ConversionTarget target(getContext());
        target.addLegalDialect<math::MathDialect>();
        target.addDynamicallyLegalOp<CallOp>([](CallOp op) {
            // Mark CallOp as legal if it's not a math intrinsic
            if (op.getCallType() != CallType::PureIntrinsic &&
                op.getCallType() != CallType::Intrinsic) {
                return true;
            }

            // Check if it's a math intrinsic we want to convert
            StringRef name = op.getName();
            static const llvm::StringSet mathIntrinsics = {"sqrt",
                                                           "sqrt_f32",
                                                           "sqrt_f64",
                                                           "sin",
                                                           "sin_f32",
                                                           "sin_f64",
                                                           "cos",
                                                           "cos_f32",
                                                           "cos_f64",
                                                           "tan",
                                                           "tan_f32",
                                                           "tan_f64",
                                                           "exp",
                                                           "exp_f32",
                                                           "exp_f64",
                                                           "exp2",
                                                           "exp2_f32",
                                                           "exp2_f64",
                                                           "log",
                                                           "log_f32",
                                                           "log_f64",
                                                           "log2",
                                                           "log2_f32",
                                                           "log2_f64",
                                                           "log10",
                                                           "log10_f32",
                                                           "log10_f64",
                                                           "fabs",
                                                           "abs_f32",
                                                           "abs_f64",
                                                           "floor",
                                                           "floor_f32",
                                                           "floor_f64",
                                                           "ceil",
                                                           "ceil_f32",
                                                           "ceil_f64",
                                                           "round",
                                                           "round_f32",
                                                           "round_f64",
                                                           "trunc",
                                                           "trunc_f32",
                                                           "trunc_f64",
                                                           "asin",
                                                           "asin_f32",
                                                           "asin_f64",
                                                           "acos",
                                                           "acos_f32",
                                                           "acos_f64",
                                                           "atan",
                                                           "atan_f32",
                                                           "atan_f64",
                                                           "atan2",
                                                           "atan2_f32",
                                                           "atan2_f64",
                                                           "sinh",
                                                           "sinh_f32",
                                                           "sinh_f64",
                                                           "cosh",
                                                           "cosh_f32",
                                                           "cosh_f64",
                                                           "tanh",
                                                           "tanh_f32",
                                                           "tanh_f64",
                                                           "erf",
                                                           "erf_f32",
                                                           "erf_f64",
                                                           "pow",
                                                           "pow_f32",
                                                           "pow_f64",
                                                           "rsqrt",
                                                           "rsqrt_f32",
                                                           "rsqrt_f64",
                                                           "fast_inverse_sqrt",
                                                           "copysign",
                                                           "copysign_f32",
                                                           "copysign_f64",
                                                           "fma",
                                                           "fma_f32",
                                                           "fma_f64"};

            // If it's a math intrinsic, mark as illegal so it gets converted
            return !mathIntrinsics.contains(name);
        });

        if (failed(applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
            return signalPassFailure();
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
    patterns.add<UnaryMathOpConversion, BinaryMathOpConversion,
                 TernaryMathOpConversion>(patterns.getContext());
}
} // namespace mlir::halide
