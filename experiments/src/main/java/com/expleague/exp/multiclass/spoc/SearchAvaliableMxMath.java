package com.expleague.exp.multiclass.spoc;

import com.expleague.commons.func.types.TypeConverter;
import com.expleague.commons.math.io.Mx2CharSequenceConversionPack;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.methods.multiclass.spoc.AbstractCodingMatrixLearning;
import com.expleague.ml.methods.multiclass.spoc.CMLHelper;
import com.expleague.ml.methods.multiclass.spoc.impl.CodingMatrixLearning;
import org.apache.commons.cli.MissingArgumentException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
  private static final Logger log = LoggerFactory.getLogger(SearchAvaliableMxMath.class);

  public static void main(String[] args) throws Exception {
    if (args.length < 1) {
      throw new MissingArgumentException("Enter the path to mx S");
    }

    final String path = args[0];
    final int l = Integer.parseInt(args[1]);
    final String[] stepsStr = Arrays.copyOfRange(args, 2, args.length);
    final Double[] steps = ArrayTools.map(stepsStr, Double.class, Double::valueOf);
    final Mx S = loadMxFromFile(path);
    findParameters(S, l, steps);
  }

  public static void findParameters(final Mx S, final int l, Double[] steps) throws Exception{
    final int k = S.rows();
//    final CodingMatrixLearning codingMatrixLearning = new CodingMatrixLearning(k, l, 0.5);

    final int units = Runtime.getRuntime().availableProcessors() - 2;
    log.info("Units: " + units);
    final ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(units, units, 30, TimeUnit.MINUTES, new LinkedBlockingDeque<>());
    for (double step : steps) {
      final double stepCopy = step;
      for (double lambdaC = 1.0; lambdaC < 1.5 * k; lambdaC += 1.0) {
        final double lambdaCCopy = lambdaC;
        threadPoolExecutor.execute(() -> {
          for (double lambdaR = 0.5; lambdaR < 3.0; lambdaR += 0.5) {
            for (double lambda1 = 1.0; lambda1 < 1.5 * k; lambda1 += 1.0) {
              final AbstractCodingMatrixLearning cml = new CodingMatrixLearning(k, l, 0.5, lambdaCCopy, lambdaR, lambda1);
              final Mx matrixB = cml.trainCodingMatrix(S);
              if (CMLHelper.checkConstraints(matrixB)) {
                synchronized (log) {
                  log.info(stepCopy + " " + lambdaCCopy + " " + lambdaR + " " + lambda1 + "\n" + matrixB.toString() + "\n");
                }
              }
            }
          }
          log.info("step" + stepCopy + " is finished");
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
