#include "mlir/Conversion/HalideToSCF/HalideToSCF.hh"

#include "mlir/Dialect/Halide/IR/HalideOps.hh"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_DEF_CONVERTHALIDETOSCF
#include "mlir/Conversion/Conversions.h.inc"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// ForOp Conversion
//===----------------------------------------------------------------------===//

struct SerialForOpConversion : OpConversionPattern<halide::ForOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(halide::ForOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        // Only Serial for
        if (op.getForType() != halide::ForType::Serial) {
            return failure();
        }
        // Get loop bounds
        Value lowerBound = adaptor.getMin();
        Value extent = adaptor.getExtent();

        // Calculate upper bound:  min + extent
        Value upperBound =
            rewriter.create<arith::AddIOp>(op.getLoc(), lowerBound, extent);

        // Create constant step of 1
        Value step = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);

        // Convert bounds to index type if needed
        Type indexType = rewriter.getIndexType();
        if (lowerBound.getType() != indexType) {
            lowerBound = rewriter.create<arith::IndexCastOp>(
                op.getLoc(), indexType, lowerBound);
        }
        if (upperBound.getType() != indexType) {
            upperBound = rewriter.create<arith::IndexCastOp>(
                op.getLoc(), indexType, upperBound);
        }

        // Create scf.for operation
        auto scfFor = rewriter.create<scf::ForOp>(op.getLoc(), lowerBound,
                                                  upperBound, step);

        // Move the body of halide.for to scf.for
        Block &halideBody = op.getBody().front();
        Block &scfBody = scfFor.getRegion().front();

        // Need to replace all variable with the arg
        // Assuming no shadowing
        rewriter.setInsertionPointToStart(&scfBody);
        auto ind = rewriter.create<arith::IndexCastOp>(
            op.getLoc(), adaptor.getMin().getType(), scfFor.getInductionVar());
        halideBody.walk([&](halide::VariableOp variable) {
            if (variable.getName() == op.getName()) {
                rewriter.setInsertionPoint(variable);
                rewriter.replaceOp(variable, ind);
            }
        });

        // Erase the existing terminator in scf.for
        rewriter.eraseOp(halideBody.getTerminator());

        // Map the halide loop variable to scf loop variable
        // The halide loop var is accessed via the name, but scf uses block args
        // We need to add the scf block argument as a mapping

        // Inline the halide body into scf body
        rewriter.inlineBlockBefore(&halideBody, &scfBody,
                                   scfBody.getTerminator()->getIterator());

        rewriter.replaceOp(op, scfFor);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// IfOp Conversion
//===----------------------------------------------------------------------===//

struct IfOpConversion : OpConversionPattern<halide::IfOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(halide::IfOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        // Create scf.if operation
        auto scfIf = rewriter.create<scf::IfOp>(
            op.getLoc(),
            /*resultTypes=*/TypeRange{}, adaptor.getCondition(),
            /*addThenBlock=*/true,
            /*addElseBlock=*/!op.getElseRegion().empty());

        auto *thenBlock = &op.getThenRegion().front();
        auto *scfThenBlock = scfIf.thenBlock();
        rewriter.inlineBlockBefore(thenBlock, scfThenBlock,
                                   scfThenBlock->end());

        // Convert halide.yield to scf.yield in then region
        auto thenYield =
            dyn_cast<halide::YieldOp>(scfThenBlock->getTerminator());
        rewriter.setInsertionPoint(thenYield);
        rewriter.replaceOpWithNewOp<scf::YieldOp>(thenYield);

        // Move else region if it exists
        if (!op.getElseRegion().empty()) {
            auto *elseBlock = &op.getElseRegion().front();
            auto *scfElseBlock = scfIf.elseBlock();
            rewriter.inlineBlockBefore(elseBlock, scfElseBlock,
                                       scfElseBlock->end());

            // Convert halide.yield to scf.yield in else region
            auto elseYield =
                dyn_cast<halide::YieldOp>(scfElseBlock->getTerminator());
            rewriter.setInsertionPoint(elseYield);
            rewriter.replaceOpWithNewOp<scf::YieldOp>(elseYield);
        }

        rewriter.replaceOp(op, scfIf);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ConvertHalideToSCFPass
    : ::impl::ConvertHalideToSCFBase<ConvertHalideToSCFPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        halide::populateHalideToSCFConversionPatterns(patterns);

        ConversionTarget target(getContext());
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<arith::ArithDialect>();

        // Mark Halide control flow ops as illegal
        target.addIllegalOp<halide::ForOp>();
        target.addIllegalOp<halide::IfOp>();

        // YieldOp is legal only when not inside Halide control flow
        target.addDynamicallyLegalOp<halide::YieldOp>([](halide::YieldOp op) {
            // Legal if parent is not a Halide control flow op being converted
            Operation *parent = op->getParentOp();
            return !isa<halide::ForOp, halide::IfOp>(parent);
        });

        if (failed(applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertHalideToSCF() {
    return std::make_unique<ConvertHalideToSCFPass>();
}
} // namespace mlir

namespace mlir::halide {
void populateHalideToSCFConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<SerialForOpConversion, IfOpConversion>(patterns.getContext());
}
} // namespace mlir::halide
