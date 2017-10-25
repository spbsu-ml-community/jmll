package com.expleague.expedia.features;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.expedia.utils.FastRandom;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class Factor {
  public static final String FILE_NAME = "factor.gz";

  private static final int EXP_COUNT = 100;
  private static final int HOTELS_COUNT = 2 * 100;

  private final VecBuilder value = new VecBuilder();

  private final FastRandom random = new FastRandom();

  // TODO: replace HashMap with fast HashMap
  // TODO: replace int[] with SparseVec
  private HashMap<Integer, int[]> users = new HashMap<>();
  private int[] hotelsTotal = new int[HOTELS_COUNT];
  private int[][] hotels = new int[HOTELS_COUNT][HOTELS_COUNT];

  // buffers
  private double[] results = new double[EXP_COUNT];
  private double[] dirichlet = new double[HOTELS_COUNT];

  public Factor() {
    Arrays.fill(hotelsTotal, HOTELS_COUNT);
    for (int[] values : hotels) {
      Arrays.fill(values, 1);
    }
  }

  public Vec build() {
    return value.build();
  }

  public void write(final String path) throws IOException {
    final ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(path + FILE_NAME)));
    out.writeObject(users);
    out.writeObject(hotelsTotal);
    out.writeObject(hotels);
    out.close();
  }

  public static Factor load(final String path) throws IOException, ClassNotFoundException {
    final Factor factor = new Factor();

    final ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(path + FILE_NAME)));
    factor.users = (HashMap<Integer, int[]>) in.readObject();
    factor.hotelsTotal = (int[]) in.readObject();
    factor.hotels = (int[][]) in.readObject();
    in.close();

    return factor;
  }

  public void addFactor(final int user, final int hotel, final int hasBooked) {
    final int key = 2 * hotel + hasBooked;

    final int[] values = getUser(user);

    // clear results
    Arrays.fill(results, 0);

    for (int expIndex = 0; expIndex < EXP_COUNT; ++expIndex) {
      random.nextDirichlet(values, dirichlet);

      for (int i = 0; i < HOTELS_COUNT; ++i) {
        results[expIndex] += dirichlet[i] * random.nextBeta(hotels[i][key], hotelsTotal[i] - hotels[i][key]);
      }
    }

    value.append(getUCB(results));
  }

  public void add(final int user, final int hotel, final int hasBooked) {
    addFactor(user, hotel, hasBooked);
    update(user, hotel, hasBooked);
  }

  private void update(final int user, final int hotel, final int hasBooked) {
    final int key = 2 * hotel + hasBooked;

    int[] values = getUser(user);

    for (int i = 0; i < HOTELS_COUNT; ++i) {
      if (values[i] > 1) {
        ++hotels[i][key];
        ++hotelsTotal[i];
      }
    }

    ++values[key];
  }

  private double getUCB(final double[] values) {
    final double average = Arrays.stream(values).average().orElse(0);
    return Math.sqrt(Arrays.stream(values).map((x) -> Math.pow(x - average, 2)).average().orElse(0));
  }

  private int[] getUser(final int user) {
    int[] values = users.get(user);

    if (values == null) {
      values = new int[HOTELS_COUNT];
      Arrays.fill(values, 1);
      users.put(user, values);
    }

    return values;
  }
}
