package com.spbsu.exp;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.logging.Interval;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.methods.PGMEM;
import com.spbsu.ml.models.pgm.ProbabilisticGraphicalModel;
import com.spbsu.ml.models.pgm.Route;
import com.spbsu.ml.models.pgm.SimplePGM;
import com.spbsu.ml.testUtils.TestResourceLoader;
import junit.framework.TestCase;

import java.io.*;
import java.util.ArrayList;
import java.util.zip.GZIPInputStream;

/**
 * Created by inikifor on 12.04.14.
 */
public class PGMEMLogDataTest extends TestCase {
  private static final int SIZE = 11;

  private FastRandom rng;
  private Action<Pair<Integer, Double>> modelValidationListener = new Action<Pair<Integer, Double>>() {

    @Override
    public void invoke(Pair<Integer, Double> data) {
      System.out.println("For top = " + data.first + " average probability: " + data.second);
    }
  };
  private LogsData learn;
  private LogsData validate;

  protected void setUp() throws Exception {
    rng = new FastRandom(0);
    learn = new LogsData(new GZIPInputStream(TestResourceLoader.loadResourceAsStream("pgmem/ses_100k_simple_rand1.dat.gz")));
    validate = new LogsData(new GZIPInputStream(TestResourceLoader.loadResourceAsStream("pgmem/ses_100k_simple_rand2.dat.gz")));
  }

  public void testMostProbable() throws IOException {
    VecDataSet dataSet = new VecDataSetImpl(new RowsVecArrayMx(learn.getRoutes()), null);
    ProbabilisticGraphicalModel model = getModel(dataSet, 100, PGMEM.MOST_PROBABLE_PATH, true);

    checkModel(model, 3, modelValidationListener);
  }

  public void testLaplacePrior() throws IOException {
    VecDataSet dataSet = new VecDataSetImpl(new RowsVecArrayMx(learn.getRoutes()), null);
    ProbabilisticGraphicalModel model = getModel(dataSet, 100, PGMEM.LAPLACE_PRIOR_PATH, true);

    checkModel(model, 3, modelValidationListener);
  }

  public void testFreqDensityPrior() throws IOException {
    VecDataSet dataSet = new VecDataSetImpl(new RowsVecArrayMx(learn.getRoutes()), null);
    ProbabilisticGraphicalModel model = getModel(dataSet, 100, PGMEM.FREQ_DENSITY_PRIOR_PATH, true);

    checkModel(model, 3, modelValidationListener);
  }

  public void testMostProbablePartial() throws IOException {
    testPartial(learn, 10, 100, PGMEM.MOST_PROBABLE_PATH);
  }

  public void testLaplacePriorPartial() throws IOException {
    testPartial(learn, 10, 100, PGMEM.LAPLACE_PRIOR_PATH);
  }

  public void testFreqDensityPriorPartial() throws IOException {
    testPartial(learn, 10, 100, PGMEM.FREQ_DENSITY_PRIOR_PATH);
  }

  private void testPartial(LogsData ld, int partitionCount, int iterations, Computable<ProbabilisticGraphicalModel, PGMEM.Policy> policy) throws IOException {
    int stepSize = ld.getRoutes().length / partitionCount;
    for (int i = 1; i <= partitionCount; i++) {
      Vec[] part = new Vec[i == partitionCount ? ld.getRoutes().length : i * stepSize];
      System.arraycopy(ld.getRoutes(), 0, part, 0, part.length);
      VecDataSet dataSet = new VecDataSetImpl(new RowsVecArrayMx(part), null);
      System.out.println("\nData set size: " + dataSet.length() + ":");
      ProbabilisticGraphicalModel model = getModel(dataSet, iterations, policy, false);
      checkModel(model, 3, modelValidationListener);
    }
  }

  private ProbabilisticGraphicalModel getModel(VecDataSet dataSet, int iterations, Computable<ProbabilisticGraphicalModel, PGMEM.Policy> policy, boolean listen)
      throws IOException {
    final Mx original = new VecBasedMx(SIZE, VecTools.fill(new ArrayVec(SIZE * SIZE), 1.));
    PGMEM pgmem = new PGMEM(original, 0.2, iterations, rng, policy);
    if (listen) {
      final Action<SimplePGM> listener = new Action<SimplePGM>() {
        int iteration = 0;

        @Override
        public void invoke(SimplePGM pgm) {
          Interval.stopAndPrint("Iteration " + ++iteration);
          System.out.println();
          System.out.print(VecTools.distance(pgm.topology, original));
          for (int i = 0; i < pgm.topology.columns(); i++) {
            System.out.print(" " + VecTools.distance(pgm.topology.row(i), original.row(i)));
          }
          System.out.println();
          Interval.start();
        }
      };
      pgmem.addListener(listener);
      Interval.start();
    }
    SimplePGM fit = pgmem.fit(dataSet, new LLLogit(VecTools.fill(new ArrayVec(dataSet.length()), 1.), dataSet));
    VecTools.fill(fit.topology.row(fit.topology.rows() - 1), 0);
    System.out.println(MxTools.prettyPrint(fit.topology));
    return fit;
  }

  private void checkModel(ProbabilisticGraphicalModel model, int accuracyLimit, Action<Pair<Integer, Double>> listener) throws IOException {
    VecDataSet check = new VecDataSetImpl(new RowsVecArrayMx(validate.getRoutes()), null);
    for (int i = 0; i < accuracyLimit; i++) {
      listener.invoke(Pair.create(i, checkModel(check, (SimplePGM) model, i)));
    }
  }

  private Route[] knownRoutes(SimplePGM model) {
    ArrayList<Route> routes = new ArrayList<Route>();
    for(int i=0; i<model.knownRoutesCount(); i++) {
      routes.add(model.knownRoute(i));
    }
    return routes.toArray(new Route[0]);
  }

  private double checkModel(DataSet check, SimplePGM fit, int accuracy) {
    final int[][] cpds = new int[check.length()][];
    final Mx data = ((VecDataSet) check).data();
    for (int j = 0; j < data.rows(); j++) {
      cpds[j] = fit.extractControlPoints(data.row(j));
    }
    Route[] knownRoutes = knownRoutes(fit);
    int count = 0;
    for (int i = 0; i < cpds.length; i++) {
      for (int j = 0; j <= Math.min(knownRoutes.length - 1, accuracy); j++) {
        if (checkRoute(knownRoutes[j], cpds[i])) {
          count++;
          break;
        }
      }
    }
    return ((double) count) / cpds.length;
  }

  private boolean checkRoute(Route route, int... controlPoints) {
    int index = 0;
    for (int t = 0; t < route.length() && index < controlPoints.length; t++) {
      if (route.dst(t) == controlPoints[index])
        index++;
    }
    return index == controlPoints.length;
  }

  public static final class LogsData {

    private static final int TYPE_OTHER = 0;
    private static final int TYPE_DIRECT = 1;
    private static final int TYPE_IMAGES = 2;
    private static final int TYPE_NEWS = 3;
    private static final int TYPE_VIDEO = 4;
    private static final int TYPE_WEB = 5;

    private static final int REL_NOT_ASED = 0;
    private static final int REL_IRRELEVANT = 1;
    private static final int REL_RELEVANT_MINUS = 2;
    private static final int REL_RELEVANT_PLUS = 3;
    private static final int REL_USEFUL = 4;
    private static final int REL_VITAL = 5;

    private Vec[] routes = new Vec[0];
    private Vec[] relevances = new Vec[0];
    private Vec[] types = new Vec[0];

    public LogsData(InputStream in) throws IOException {
      BufferedReader reader = null;
      ArrayList<Vec> lrouts = new ArrayList<Vec>();
      ArrayList<Vec> lrels = new ArrayList<Vec>();
      ArrayList<Vec> ltypes = new ArrayList<Vec>();
      try {
        reader = new BufferedReader(new InputStreamReader(in));
        String line;
        while ((line = reader.readLine()) != null) {
          JsonObject obj = new JsonParser().parse(line.split("\t")[1]).getAsJsonObject();
          JsonArray clicks = obj.getAsJsonArray("clicks");
          Vec vclicks = new ArrayVec(0);
          if (clicks != null && !clicks.isJsonNull()) {
            vclicks = new ArrayVec(clicks.size() + 1);
            int i = 0;
            for(JsonElement el: clicks) {
              vclicks.set(i++, el.getAsInt() + 1);
            }
          }
          lrouts.add(vclicks);
          Vec vrel = new ArrayVec(0);
          JsonArray rels = obj.getAsJsonArray("rel");
          if (rels != null && !rels.isJsonNull()) {
            vrel = new ArrayVec(rels.size());
            int i = 0;
            for(JsonElement el: rels) {
              vrel.set(i++, parseRelevance(el.getAsString()));
            }
          }
          lrels.add(vrel);
          Vec vtype = new ArrayVec(0);
          JsonArray types = obj.getAsJsonArray("sntypes");
          if (types != null && !types.isJsonNull()) {
            vtype = new ArrayVec(types.size());
            int i = 0;
            for(JsonElement el: types) {
              vtype.set(i++, parseType(el.getAsString()));
            }
          }
          ltypes.add(vtype);
        }
        routes = lrouts.toArray(routes);
        relevances = lrels.toArray(relevances);
        types = ltypes.toArray(types);
      } finally {
        if (reader != null) {
          reader.close();
        }
      }
    }

    private int parseType(String type) {
      int result = TYPE_OTHER;
      if (type.equals("DIRECT")) {
        result = TYPE_DIRECT;
      }
      if (type.equals("IMAGES")) {
        result = TYPE_IMAGES;
      }
      if (type.equals("NEWS")) {
        result = TYPE_NEWS;
      }
      if (type.equals("VIDEO")) {
        result = TYPE_VIDEO;
      }
      if (type.equals("WEB")) {
        result = TYPE_WEB;
      }
      return result;
    }

    private int parseRelevance(String relevance) {
      int result = REL_NOT_ASED;
      if (relevance.equals("IRRELEVANT")) {
        result = REL_IRRELEVANT;
      }
      if (relevance.equals("RELEVANT_MINUS")) {
        result = REL_RELEVANT_MINUS;
      }
      if (relevance.equals("RELEVANT_PLUS")) {
        result = REL_RELEVANT_PLUS;
      }
      if (relevance.equals("USEFUL")) {
        result = REL_USEFUL;
      }
      if (relevance.equals("VITAL")) {
        result = REL_VITAL;
      }
      return result;
    }

    public Vec[] getRoutes() {
      return routes;
    }

    public Vec[] getRelevances() {
      return relevances;
    }

    public Vec[] getTypes() {
      return types;
    }
  }

}
