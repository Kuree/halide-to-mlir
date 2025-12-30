#include "mlir/Target/Halide/ImportHalide.hh"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Halide/IR/HalideOps.hh"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "import-halide"

namespace {

using namespace Halide::Internal;
using namespace mlir;

// Helper to convert Halide Types to MLIR Types
Type convertType(OpBuilder &builder, Halide::Type t) {
    if (t.is_float()) {
        if (t.bits() == 16)
            return builder.getF16Type();
        if (t.bits() == 32)
            return builder.getF32Type();
        if (t.bits() == 64)
            return builder.getF64Type();
    } else if (t.is_int() || t.is_uint()) {
        // MLIR standard integers are signless.
        // We might want to use signed/unsigned ops based on Halide type,
        // but the storage type is just IntegerType.
        return builder.getIntegerType(t.bits());
    } else if (t.is_handle()) {
        // Represent handles using the Halide handle type
        return builder.getType<halide::HandleType>();
    } else if (t.is_bool()) {
        return builder.getI1Type();
    }
    // Vector types
    if (t.lanes() > 1) {
        Type elemType = convertType(builder, t.element_of());
        return VectorType::get({t.lanes()}, elemType);
    }
    return builder.getNoneType();
}

class HalideToMLIRVisitor : public IRVisitor {
  public:
    OpBuilder &builder;
    std::stack<Value> valueStack;

    HalideToMLIRVisitor(OpBuilder &b) : builder(b) {}

    void pushValue(Value v) { valueStack.push(v); }

    Value popValue() {
        assert(!valueStack.empty() && "Value stack underflow");
        Value v = valueStack.top();
        valueStack.pop();
        return v;
    }

    // Helper to visit a stmt and wrap it in a region
    void createRegionBody(Region &region, const Stmt &bodyStmt) {
        auto *block = builder.createBlock(&region);
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToEnd(block);

            if (bodyStmt.defined()) {
                bodyStmt.accept(this);
            }

            builder.setInsertionPointToEnd(block);
            builder.create<halide::YieldOp>(builder.getUnknownLoc());
        }
    }

    // --- Expr Visitors ---

    void visit(const IntImm *op) override {
        Type t = convertType(builder, op->type);
        Value v = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(),
                                                       op->value, t);
        pushValue(v);
    }

    void visit(const UIntImm *op) override {
        Type t = convertType(builder, op->type);
        // Arith constant int op handles both signed and unsigned via API bits
        Value v = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(),
                                                       op->value, t);
        pushValue(v);
    }

    void visit(const FloatImm *op) override {
        Type t = convertType(builder, op->type);
        Value v = builder.create<arith::ConstantFloatOp>(
            builder.getUnknownLoc(), APFloat(op->value), cast<FloatType>(t));
        pushValue(v);
    }

    void visit(const StringImm *op) override {
        // use llvm global
        auto *insertionBlock = builder.getInsertionBlock();
        auto *symTabOp =
            SymbolTable::getNearestSymbolTable(insertionBlock->getParentOp());
        SymbolTable symTab(symTabOp);
        auto const &val = op->value;
        auto type = LLVM::LLVMArrayType::get(builder.getI8Type(), val.size());
        LLVM::GlobalOp global;
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToEnd(&symTabOp->getRegion(0).back());
            global = builder.create<LLVM::GlobalOp>(
                builder.getUnknownLoc(), type,
                /*isConstant=*/true, LLVM::Linkage::Private, "str",
                builder.getStringAttr(val), /* alignment */ 0);

            symTab.insert(global);
        }

        // create a ptr
        auto addressOf =
            builder.create<LLVM::AddressOfOp>(builder.getUnknownLoc(), global);
        pushValue(addressOf);
    }

    void visit(const Variable *op) override {
        Value newVar = builder.create<halide::VariableOp>(
            builder.getUnknownLoc(), convertType(builder, op->type),
            builder.getStringAttr(op->name));
        pushValue(newVar);
    }

    void visit(const Cast *op) override {
        op->value.accept(this);
        Value val = popValue();
        Value res = builder.create<halide::CastOp>(
            builder.getUnknownLoc(), convertType(builder, op->type), val);
        pushValue(res);
    }

    // Binary Arithmetic Helper
    template <typename OpType>
    void visitBinaryOp(const Halide::Expr &a, const Halide::Expr &b) {
        a.accept(this);
        Value lhs = popValue();
        b.accept(this);
        Value rhs = popValue();

        // handle !halide.handle type comparison without a cast op
        auto lhsType = lhs.getType();
        auto rhsType = rhs.getType();
        if (isa<halide::HandleType>(lhsType) && rhsType.isIntOrFloat()) {
            lhs = builder.create<halide::CastOp>(builder.getUnknownLoc(),
                                                 rhsType, lhs);
        } else if (isa<halide::HandleType>(rhsType) && lhsType.isIntOrFloat()) {
            rhs = builder.create<halide::CastOp>(builder.getUnknownLoc(),
                                                 lhsType, rhs);
        }

        Value res = builder.create<OpType>(builder.getUnknownLoc(), lhs, rhs);
        pushValue(res);
    }

    void visit(const Add *op) override {
        visitBinaryOp<halide::AddOp>(op->a, op->b);
    }
    void visit(const Sub *op) override {
        visitBinaryOp<halide::SubOp>(op->a, op->b);
    }
    void visit(const Mul *op) override {
        visitBinaryOp<halide::MulOp>(op->a, op->b);
    }
    void visit(const Div *op) override {
        visitBinaryOp<halide::DivOp>(op->a, op->b);
    }
    void visit(const Mod *op) override {
        visitBinaryOp<halide::ModOp>(op->a, op->b);
    }
    void visit(const Min *op) override {
        visitBinaryOp<halide::MinOp>(op->a, op->b);
    }
    void visit(const Max *op) override {
        visitBinaryOp<halide::MaxOp>(op->a, op->b);
    }

    void visit(const EQ *op) override {
        visitBinaryOp<halide::EQOp>(op->a, op->b);
    }
    void visit(const NE *op) override {
        visitBinaryOp<halide::NEOp>(op->a, op->b);
    }
    void visit(const LT *op) override {
        visitBinaryOp<halide::LTOp>(op->a, op->b);
    }
    void visit(const LE *op) override {
        visitBinaryOp<halide::LEOp>(op->a, op->b);
    }
    void visit(const GT *op) override {
        visitBinaryOp<halide::GTOp>(op->a, op->b);
    }
    void visit(const GE *op) override {
        visitBinaryOp<halide::GEOp>(op->a, op->b);
    }

    void visit(const And *op) override {
        visitBinaryOp<halide::AndOp>(op->a, op->b);
    }
    void visit(const Or *op) override {
        visitBinaryOp<halide::OrOp>(op->a, op->b);
    }

    void visit(const Not *op) override {
        op->a.accept(this);
        Value val = popValue();
        Value res = builder.create<halide::NotOp>(builder.getUnknownLoc(), val);
        pushValue(res);
    }

    void visit(const Select *op) override {
        op->condition.accept(this);
        Value cond = popValue();
        op->true_value.accept(this);
        Value tVal = popValue();
        op->false_value.accept(this);
        Value fVal = popValue();

        // Stack order: cond, true, false -> pop: false, true, cond
        // Wait, standard visit order: cond, true, false.
        // Stack: [cond], [cond, true], [cond, true, false].
        // Pop: false. Pop: true. Pop: cond.
        // So reverse order of popping is required.
        std::swap(tVal, fVal); // fVal is now actually true_value
        std::swap(cond, fVal); // cond is now actually condition, tVal is false,
                               // fVal is true (wait, logic hard)

        // Correct order:
        // 1. visit(cond) -> push(cond)
        // 2. visit(true) -> push(true)
        // 3. visit(false) -> push(false)
        // Stack top: false.

        // Let's redo logic cleanly:
        // Value valFalse = popValue();
        // Value valTrue = popValue();
        // Value valCond = popValue();

        // The implementation above with swaps was messy.
        // Correct logic:
        Value falseV = popValue();
        Value trueV = popValue();
        Value condV = popValue();

        Value res = builder.create<halide::SelectOp>(builder.getUnknownLoc(),
                                                     condV, trueV, falseV);
        pushValue(res);
    }

    void visit(const Load *op) override {
        op->index.accept(this);
        Value idx = popValue();
        op->predicate.accept(this);
        Value pred = popValue();

        // Since predicate is usually on top if visited second.
        // Just be consistent with visit order.

        Value res = builder.create<halide::LoadOp>(
            builder.getUnknownLoc(), convertType(builder, op->type),
            builder.getStringAttr(op->name), idx, pred);
        pushValue(res);
    }

    void visit(const Call *op) override {
        // Collect args
        std::vector<Value> args;
        for (const auto &arg : op->args) {
            arg.accept(this);
            args.push_back(popValue());
        }

        // Create the CallOp with the function name, call type, and arguments
        Value res = builder.create<halide::CallOp>(
            builder.getUnknownLoc(), convertType(builder, op->type),
            builder.getStringAttr(op->name),
            static_cast<halide::CallType>(op->call_type), args);

        pushValue(res);
    }

    void visit(const Ramp *op) override { llvm_unreachable("Not implemented"); }

    void visit(const Broadcast *op) override {
        llvm_unreachable("Not implemented");
    }

    void visit(const Let *op) override {
        op->value.accept(this);
        Value val = popValue();

        op->body.accept(this);
    }

    // --- Stmt Visitors ---

    void visit(const LetStmt *op) override {
        op->value.accept(this);
        Value val = popValue();

        auto letOp = builder.create<halide::LetStmtOp>(
            builder.getUnknownLoc(), builder.getStringAttr(op->name), val);

        Region &region = letOp.getBody();
        auto *block = builder.createBlock(&region);
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToEnd(block);

            if (op->body.defined())
                op->body.accept(this);

            builder.setInsertionPointToEnd(block);
            builder.create<halide::YieldOp>(
                NameLoc::get(builder.getStringAttr(op->name)));
        }
    }

    void visit(const AssertStmt *op) override {
        op->condition.accept(this);
        Value cond = popValue();

        auto assertOp =
            builder.create<halide::AssertStmtOp>(builder.getUnknownLoc(), cond);

        OpBuilder::InsertionGuard guard(builder);
        Region &region = assertOp.getMessage();
        auto *block = builder.createBlock(&region);
        builder.setInsertionPointToEnd(block);

        // Visit the message expression
        op->message.accept(this);
        Value msg = popValue();

        // Yield the message value
        builder.setInsertionPointToEnd(block);
        builder.create<halide::YieldOp>(builder.getUnknownLoc());
    }

    void visit(const ProducerConsumer *op) override {
        auto producerConsumerOp = builder.create<halide::ProducerConsumerOp>(
            builder.getUnknownLoc(), builder.getStringAttr(op->name),
            builder.getBoolAttr(op->is_producer));

        Region &region = producerConsumerOp.getBody();
        auto *block = builder.createBlock(&region);
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToEnd(block);

            if (op->body.defined())
                op->body.accept(this);

            builder.setInsertionPointToEnd(block);
            builder.create<halide::YieldOp>(builder.getUnknownLoc());
        }
    }

    void visit(const For *op) override {
        op->min.accept(this);
        Value minVal = popValue();
        op->extent.accept(this);
        Value extentVal = popValue();

        auto forOp = builder.create<halide::ForOp>(
            builder.getUnknownLoc(), builder.getStringAttr(op->name), minVal,
            extentVal, static_cast<halide::ForType>(op->for_type),
            static_cast<halide::DeviceAPI>(op->device_api),
            static_cast<halide::Partition>(op->partition_policy));

        // Loop variable needs to be defined in the body.
        // Halide For loops implicitly define the loop variable `op->name`.
        // In MLIR `scf.for`, the IV is a block argument.
        // We should add a block argument for the induction variable.

        Region &region = forOp.getBody();
        auto *block = builder.createBlock(&region);

        {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToEnd(block);

            if (op->body.defined())
                op->body.accept(this);
            builder.setInsertionPointToEnd(block);
            builder.create<halide::YieldOp>(
                NameLoc::get(builder.getStringAttr(op->name)));
        }
    }

    void visit(const Store *op) override {
        op->value.accept(this);
        Value val = popValue();
        op->index.accept(this);
        Value idx = popValue();
        op->predicate.accept(this);
        Value pred = popValue();

        builder.create<halide::StoreOp>(builder.getUnknownLoc(),
                                        builder.getStringAttr(op->name), val,
                                        idx, pred);
    }

    void visit(const Provide *op) override {
        llvm_unreachable("Not implemented");
    }

    void visit(const Allocate *op) override {
        op->condition.accept(this);
        Value cond = popValue();

        std::vector<Value> extents;
        for (const auto &e : op->extents) {
            e.accept(this);
            extents.push_back(popValue());
        }

        auto allocOp = builder.create<halide::AllocateOp>(
            builder.getUnknownLoc(), op->name, convertType(builder, op->type),
            static_cast<halide::MemoryType>(op->memory_type), extents, cond);

        createRegionBody(allocOp.getBody(), op->body);
    }

    void visit(const Realize *op) override {
        // Similar to Allocate but for Funcs.
        // We can treat it as a scope or ignore if lowering handles it.
        // Assuming we visit body.
        op->body.accept(this);
    }

    void visit(const Halide::Internal::Block *op) override {
        op->first.accept(this);
        if (op->rest.defined()) {
            op->rest.accept(this);
        }
    }

    void visit(const IfThenElse *op) override {
        op->condition.accept(this);
        Value cond = popValue();

        auto ifOp = builder.create<halide::IfOp>(builder.getUnknownLoc(), cond);

        OpBuilder::InsertionGuard guard(builder);

        createRegionBody(ifOp.getThenRegion(), op->then_case);

        createRegionBody(ifOp.getElseRegion(), op->else_case);
    }

    void visit(const Evaluate *op) override {
        op->value.accept(this);
        Value v = popValue();
        // Evaluate just computes the value for side effects.
        // In MLIR, the operations generated by 'accept' are inserted into the
        // block. We just discard the result value from the stack.
        (void)v;
    }

    void visit(const Shuffle *op) override {
        llvm::errs() << "Shuffle not impl\n";
    }
    void visit(const Prefetch *op) override {
        llvm::errs() << "Prefetch not impl\n";
    }
    void visit(const Atomic *op) override {
        llvm_unreachable("Not implemented");
    }
    void visit(const Free *op) override { llvm_unreachable("Not implemented"); }
    void visit(const Acquire *op) override {
        llvm_unreachable("Not implemented");
    }
    void visit(const Fork *op) override { llvm_unreachable("Not implemented"); }
};

LoweredFunc getLoweredFunc(Halide::Func func, const Halide::Target &target) {
    Halide::Module m =
        func.compile_to_module(func.infer_arguments(), func.name(), target);
    auto loweredFunc = m.get_function_by_name(func.name());
    return loweredFunc;
}

void setupFuncArgs(ArrayRef<LoweredArgument> args, func::FuncOp funcOp,
                   OpBuilder &builder) {
    // compute the new type
    SmallVector<Type> types;
    llvm::transform(args, std::back_inserter(types),
                    [&](const Halide::Argument &arg) -> Type {
                        if (arg.kind == Halide::Argument::Kind::InputScalar) {
                            return convertType(builder, arg.type);
                        }
                        return builder.getType<halide::HandleType>();
                    });
    auto newType = builder.getFunctionType(types, {});
    funcOp.setFunctionType(newType);
    auto *newBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(newBlock);
    // use let statement to set the scope properly
    for (auto const [idx, arg] : llvm::enumerate(args)) {
        auto argValue = newBlock->getArgument(idx);
        // some hack to get the proper buffer reference
        std::string name = arg.name;
        if (arg.kind != Halide::Argument::Kind::InputScalar) {
            name.append(".buffer");
        }
        auto letStmt = builder.create<halide::LetStmtOp>(
            NameLoc::get(builder.getStringAttr(arg.name)), name, argValue);
        auto *block = builder.createBlock(&letStmt.getBody());
        builder.setInsertionPointToEnd(block);
        auto yield = builder.create<halide::YieldOp>(builder.getUnknownLoc());
        builder.setInsertionPoint(yield);
    }
}

} // namespace

namespace mlir::halide {

OwningOpRef<ModuleOp> importHalide(Halide::Func func, MLIRContext *context,
                                   const Halide::Target &target) {
    auto name = func.name();
    auto loweredFunc = getLoweredFunc(std::move(func), target);
    OwningOpRef result =
        ModuleOp::create(NameLoc::get(StringAttr::get(context, name)));

    OpBuilder builder(context);
    builder.setInsertionPointToEnd(result->getBody());

    auto funcOp = builder.create<func::FuncOp>(builder.getUnknownLoc(), name,
                                               builder.getFunctionType({}, {}));
    setupFuncArgs(loweredFunc.args, funcOp, builder);

    HalideToMLIRVisitor visitor(builder);
    loweredFunc.body.accept(&visitor);
    assert(visitor.valueStack.empty() && "Incorrect visit state");
    builder.setInsertionPointToEnd(&funcOp.getBody().back());
    builder.create<func::ReturnOp>(builder.getUnknownLoc());

    return result;
}

} // namespace mlir::halide