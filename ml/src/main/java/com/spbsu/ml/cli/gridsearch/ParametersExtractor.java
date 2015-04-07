package com.spbsu.ml.cli.gridsearch;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * User: qdeee
 * Date: 25.03.15
 */
public class ParametersExtractor {
  private static final DecimalFormat formatter = new DecimalFormat("###.########", new DecimalFormatSymbols(Locale.US));

  public static String[][] parse(final String input) {
    final List<String[]> result = new ArrayList<>();
    final String[] splitParameters = input.split(";");
    for (String parameter : splitParameters) {
      result.add(parseParameter(parameter));
    }
    return result.toArray(new String[result.size()][]);
  }

  private static String[] parseParameter(final String parameterSpec) {
    final List<String> result = new ArrayList<>();
    final String[] rangesAndEnumarations = parameterSpec.split(",");
    for (String s : rangesAndEnumarations) {
      if (s.contains(":")) {
        Collections.addAll(result, parseInterval(s));
      } else {
        result.add(s);
      }
    }
    return result.toArray(new String[result.size()]);
  }

  private static String[] parseInterval(final String interval) {
    final String[] split = interval.split(":");
    if (split.length != 3) {
      throw new IllegalArgumentException("Invalid interval: " + interval);
    }

    final double start = Double.parseDouble(split[0]);
    final double end = Double.parseDouble(split[1]);
    final double step = Double.parseDouble(split[2]);

    final List<String> result = new ArrayList<>();
    for (double w = start; Math.nextUp(w) < end; w += step) {
      result.add(formatter.format(w));
    }
    return result.toArray(new String[result.size()]);
  }
}
