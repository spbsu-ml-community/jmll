package com.expleague.erc.lambda;

import java.io.Serializable;
import java.util.function.DoubleUnaryOperator;

public class LambdaTransforms {
    private LambdaTransforms() {}

    public static class IdentityTransform implements DoubleUnaryOperator, Serializable {
        @Override
        public double applyAsDouble(double v) {
            return v;
        }
    }

    public static class IdentityDerivativeTransform implements DoubleUnaryOperator, Serializable {
        @Override
        public double applyAsDouble(double v) {
            return 1;
        }
    }

    public static class AbsTransform implements DoubleUnaryOperator, Serializable {
        @Override
        public double applyAsDouble(double v) {
            return Math.abs(v);
        }
    }

    public static class AbsDerivativeTransform implements DoubleUnaryOperator, Serializable {
        @Override
        public double applyAsDouble(double v) {
            return Math.signum(v);
        }
    }

    public static class SqrTransform implements DoubleUnaryOperator, Serializable {
        @Override
        public double applyAsDouble(double v) {
            return v * v;
        }
    }

    public static class SqrDerivativeTransform implements DoubleUnaryOperator, Serializable {
        @Override
        public double applyAsDouble(double v) {
            return 2 * v;
        }
    }
}
