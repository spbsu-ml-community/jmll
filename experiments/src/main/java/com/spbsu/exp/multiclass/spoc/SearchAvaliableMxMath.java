package com.spbsu.exp.multiclass.spoc;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.math.io.Mx2CharSequenceConversionPack;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.methods.multiclass.spoc.AbstractCodingMatrixLearning;
import com.spbsu.ml.methods.multiclass.spoc.CMLHelper;
import com.spbsu.ml.methods.multiclass.spoc.impl.CodingMatrixLearning;
import org.apache.commons.cli.MissingArgumentException;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * User: qdeee
 * Date: 22.05.14
 */
public class SearchAvaliableMxMath {
  public static void main(String[] args) throws Exception {
    if (args.length < 1) {
      throw new MissingArgumentException("Enter the path to mx S");
    }

    final String path = args[0];
    final int l = Integer.parseInt(args[1]);
    final String[] stepsStr = Arrays.copyOfRange(args, 2, args.length);
    final Double[] steps = ArrayTools.map(stepsStr, Double.class, new Computable<String, Double>() {
      @Override
      public Double compute(final String argument) {
        return Double.valueOf(argument);
      }
    });
    final Mx S = loadMxFromFile(path);
    findParameters(S, l, steps);
  }

  public static void findParameters(final Mx S, final int l, Double[] steps) throws Exception{
    final Logger logger = Logger.create(SearchAvaliableMxMath.class);

    final int k = S.rows();
//    final CodingMatrixLearning codingMatrixLearning = new CodingMatrixLearning(k, l, 0.5);

    final int units = Runtime.getRuntime().availableProcessors() - 2;
    logger.info("Units: " + units);
    final ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(units, units, 30, TimeUnit.MINUTES, new LinkedBlockingDeque<Runnable>());
    for (double step : steps) {
      final double stepCopy = step;
      for (double lambdaC = 1.0; lambdaC < 1.5 * k; lambdaC += 1.0) {
        final double lambdaCCopy = lambdaC;
        threadPoolExecutor.execute(new Runnable() {
          @Override
          public void run() {
            for (double lambdaR = 0.5; lambdaR < 3.0; lambdaR += 0.5) {
              for (double lambda1 = 1.0; lambda1 < 1.5 * k; lambda1 += 1.0) {
                final AbstractCodingMatrixLearning cml = new CodingMatrixLearning(k, l, 0.5, lambdaCCopy, lambdaR, lambda1);
                final Mx matrixB = cml.trainCodingMatrix(S);
                if (CMLHelper.checkConstraints(matrixB)) {
                  synchronized (logger) {
                    logger.info(stepCopy + " " + lambdaCCopy + " " + lambdaR + " " + lambda1 + "\n" + matrixB.toString() + "\n");
                  }
                }
              }
            }
            logger.info("step" + stepCopy + " is finished");
          }
        });
      }
    }
    threadPoolExecutor.awaitTermination(24, TimeUnit.HOURS);
  }

  private static Mx loadMxFromFile(final String filename) throws IOException {
    final TypeConverter<CharSequence, Mx> converter = new Mx2CharSequenceConversionPack.CharSequence2MxConverter();
    final BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
    final StringBuilder builder = new StringBuilder();
    String s;
    while ((s = reader.readLine()) != null) {
      builder.append(s);
      builder.append("\n");
    }
    return converter.convert(builder.toString());
  }

}
