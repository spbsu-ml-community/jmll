package com.expleague.ml.distributions.parametric;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVecBuilder;
import com.expleague.ml.distributions.parametric.impl.BetaVecDistributionImpl;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.special.Beta;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

import static com.expleague.commons.math.MathTools.sqr;

public interface BetaDistribution extends RandomVariable<BetaDistribution> {

  double alpha();

  double beta();

  default RandomVecBuilder<BetaDistribution> vecBuilder() {
    return new BetaVecDistributionImpl.Builder();
  }

  BetaDistribution update(double alpha, double beta);

  public class Stub {

    public static double computeZ(double alpha, double beta) {
      return Gamma.logGamma(alpha) + Gamma.logGamma(beta) - Gamma.logGamma(alpha + beta);
    }

    public static double expectation(double alpha, double beta) {
//      final double var = (alpha * beta) / (sqr(alpha + beta) * (alpha + beta + 1));
      return (alpha ) / (alpha + beta);
//      return (alpha ) / (alpha + beta);
    }

    public static double cumulativeProbability(final double x, final double alpha, final double beta) {
      return x <= 0.0D ? 0.0D : (x >= 1.0D ? 1.0D : Beta.regularizedBeta(x, alpha, beta));
    }

    public static double instance(final FastRandom random, double alpha, double beta) {
        //TODO: is it fastest way?
        final double x = random.nextGamma(alpha);
        final double y = random.nextGamma(beta);
        return x / (x + y);
    }

    public static double logDensity(final double x, final double alpha, final double beta, final double z) {
      if (x >= 0.0D && x <= 1.0D) {
        if (x == 0.0D) {
          if (alpha < 1.0D) {
            throw new NumberIsTooSmallException(LocalizedFormats.CANNOT_COMPUTE_BETA_DENSITY_AT_0_FOR_SOME_ALPHA, Double.valueOf(alpha), Integer.valueOf(1), false);
          }
          else {
            return -1.0D / 0.0;
          }
        }
        else if (x == 1.0D) {
          if (beta < 1.0D) {
            throw new NumberIsTooSmallException(LocalizedFormats.CANNOT_COMPUTE_BETA_DENSITY_AT_1_FOR_SOME_BETA, Double.valueOf(beta), Integer.valueOf(1), false);
          }
          else {
            return -1.0D / 0.0;
          }
        }
        else {
          double logX = FastMath.log(x);
          double log1mX = FastMath.log1p(-x);
          return (alpha - 1.0D) * logX + (beta - 1.0D) * log1mX - z;
        }
      }
      else {
        return -1.0D / 0.0;
      }
    }

    public static boolean equals(final BetaDistribution left,
                                 final BetaDistribution right) {
      if (left == right) return true;

      if (Double.compare(left.alpha(), right.alpha()) != 0) return false;
      return Double.compare(left.beta(), right.beta()) == 0;
    }

    public static int hashCode(final BetaDistribution beta) {
      int result;
      long temp;
      temp = Double.doubleToLongBits(beta.alpha());
      result = (int) (temp ^ (temp >>> 32));
      temp = Double.doubleToLongBits(beta.beta());
      result = 31 * result + (int) (temp ^ (temp >>> 32));
      return result;
    }
  }
}
