package com.spbsu.exp.multiclass.spoc;

import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.math.io.Mx2CharSequenceConversionPack;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.ml.methods.multiclass.spoc.AbstractCodingMatrixLearning;
import com.spbsu.ml.methods.multiclass.spoc.impl.CodingMatrixLearningGreedyParallels;
import org.apache.commons.cli.MissingArgumentException;

import java.io.*;

/**
 * User: qdeee
 * Date: 06.06.14
 */
public class GreedyCML {
  public static void main(String[] args) throws MissingArgumentException, IOException {
    if (args.length < 1) {
      throw new MissingArgumentException("Enter the path to mx S");
    }

    final String path = args[0];
    final int l = Integer.parseInt(args[1]);
    final double lac = Double.parseDouble(args[2]);
    final double lar = Double.parseDouble(args[3]);
    final double la1 = Double.parseDouble(args[4]);

    final Mx S = loadMxFromFile(path);

    final AbstractCodingMatrixLearning cml = new CodingMatrixLearningGreedyParallels(S.rows(), l, lac, lar, la1);
    final Mx codingMatrix = cml.trainCodingMatrix(S);
    writeMxToFile(codingMatrix, "resultMxFile.txt");
  }

  private static void writeMxToFile(final Mx mx, final String filename) throws FileNotFoundException {
    final PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(filename)));
    final TypeConverter<Mx, CharSequence> converter = new Mx2CharSequenceConversionPack.Mx2CharSequenceConverter();
    final CharSequence mxStr = converter.convert(mx);
    writer.write(mxStr.toString());
    writer.flush();
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
