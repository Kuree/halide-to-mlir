#include "mlir/Conversion/HalideToFunc/HalideToFunc.hh"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Halide/IR/HalideOps.hh"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_DEF_CONVERTHALIDETOFUNC
#include "mlir/Conversion/Conversions.h.inc"

using namespace mlir;

namespace {

auto constexpr putsName = "puts";
auto constexpr abortName = "abort";

void createFuncExternIfNotExits(ModuleOp moduleOp, OpBuilder &builder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());
    if (!moduleOp.lookupSymbol<func::FuncOp>(putsName)) {
        auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
        auto i32Type = builder.getI32Type();
        auto funcType = builder.getFunctionType({ptrType}, {i32Type});

        auto funcOp =
            builder.create<func::FuncOp>(moduleOp.getLoc(), putsName, funcType);
        funcOp.setPrivate();
    }

    if (!moduleOp.lookupSymbol<func::FuncOp>(abortName)) {
        auto funcType = builder.getFunctionType({}, {});
        auto funcOp = builder.create<func::FuncOp>(moduleOp.getLoc(), abortName,
                                                   funcType);
        funcOp.setPrivate();
    }
}

//===----------------------------------------------------------------------===//
// AssertOp to func.call Conversion
//===----------------------------------------------------------------------===//

struct AssertOpConversion : OpConversionPattern<halide::AssertStmtOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(halide::AssertStmtOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto condition = adaptor.getCondition();

        // Extract the message string from the body
        std::string message;
        op.getBody()->walk([&](halide::CallOp callOp) {
            // Get the function name from the call
            message = callOp.getName();
        });

        // If no call found, use a default message
        if (message.empty()) {
            message = "assertion_failed";
        }
        message.append("\0");

        auto moduleOp = op->getParentOfType<ModuleOp>();
        createFuncExternIfNotExits(moduleOp, rewriter);

        // Create a global string constant for the message
        auto stringType =
            LLVM::LLVMArrayType::get(rewriter.getI8Type(), message.size());

        // Create a unique name for the global string
        LLVM::GlobalOp globalOp;
        {
            static auto constexpr globalName = "__assert_msg";
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(moduleOp.getBody());
            SymbolTable symbolTable(moduleOp);
            globalOp = rewriter.create<LLVM::GlobalOp>(
                loc, stringType,
                /*isConstant=*/true, LLVM::Linkage::Private, globalName,
                rewriter.getStringAttr(message),
                /*alignment=*/0);
            symbolTable.insert(globalOp);
        }

        // Create an if-then block:  if (! condition) { puts(msg); abort(); }
        // First, invert the condition
        auto i1Type = rewriter.getI1Type();
        auto trueConst = rewriter.create<arith::ConstantOp>(
            loc, i1Type, rewriter.getBoolAttr(true));
        auto invertedCondition =
            rewriter.create<arith::XOrIOp>(loc, condition, trueConst);

        // Create the if block
        auto ifOp = rewriter.create<scf::IfOp>(
            loc,
            /*resultTypes=*/TypeRange{}, invertedCondition,
            /*addThenBlock=*/true, /*addElseBlock=*/false);
        rewriter.setInsertionPointToEnd(ifOp.thenBlock());
        // Create addressof to get pointer to the global string
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto addressOf = rewriter.create<LLVM::AddressOfOp>(loc, ptrType,
                                                            globalOp.getName());
        // Call puts with the message
        rewriter.create<func::CallOp>(loc, putsName, rewriter.getI32Type(),
                                      ValueRange{addressOf});

        // Call abort
        rewriter.create<func::CallOp>(loc, abortName, TypeRange{},
                                      ValueRange{});
        // End the if block
        rewriter.create<scf::YieldOp>(loc);
        // Remove the original assert op
        rewriter.replaceOp(op, ifOp);

        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ConvertHalideToFuncPass
    : ::impl::ConvertHalideToFuncBase<ConvertHalideToFuncPass> {
    using ConvertHalideToFuncBase::ConvertHalideToFuncBase;

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        halide::populateHalideToFuncConversionPatterns(patterns);

        ConversionTarget target(getContext());
        target.addLegalDialect<func::FuncDialect, LLVM::LLVMDialect,
                               scf::SCFDialect, arith::ArithDialect>();
        target.addIllegalOp<halide::AssertStmtOp>();

        if (failed(applyPartialConversion(getOperation(), target,
                                          std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertHalideToFunc() {
    return std::make_unique<ConvertHalideToFuncPass>();
}
namespace halide {
void populateHalideToFuncConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<AssertOpConversion>(patterns.getContext());
}
} // namespace halide
} // namespace mlir
