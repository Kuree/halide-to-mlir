#include "mlir/Dialect/Halide/IR/HalideOps.hh"
#include "mlir/Dialect/Halide/Transforms/Passes.hh"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_LET2REG
#include "mlir/Dialect/Halide/Transforms/Passes.h.inc"

using namespace mlir;

namespace {

struct ReplaceLet : OpRewritePattern<halide::LetStmtOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(halide::LetStmtOp op,
                                  PatternRewriter &rewriter) const override {
        // because of potential name shadowing, we only start from
        // innermost statement
        if (!op.getBody().getOps<halide::LetStmtOp>().empty())
            return failure();
        auto val = op.getValue();

        op.walk([&](halide::VariableOp var) {
            if (var.getName() == op.getName()) {
                rewriter.replaceOp(var, val);
            }
        });

        // inline this op
        auto *block = &op.getBody().front();
        rewriter.eraseOp(block->getTerminator());
        rewriter.inlineBlockBefore(block, op->getBlock(),
                                   rewriter.getInsertionPoint());
        rewriter.eraseOp(op);
        return success();
    }
};

struct Let2RegPass : ::impl::Let2RegBase<Let2RegPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<ReplaceLet>(&getContext());
        GreedyRewriteConfig config;
        // bottom up to prevent lots of mismatches
        config.useTopDownTraversal = false;
        if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns), config)))
            return signalPassFailure();
    }
};

} // namespace

namespace mlir::halide {
std::unique_ptr<::mlir::Pass> createLet2Reg() {
    return std::make_unique<Let2RegPass>();
}
} // namespace mlir::bf
