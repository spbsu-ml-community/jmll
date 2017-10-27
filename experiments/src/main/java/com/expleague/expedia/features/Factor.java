package com.expleague.expedia.features;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.expedia.utils.FastRandom;

import java.io.*;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class Factor {
  private static final String FILE_NAME = "factor.gz";
  private static final int WORKERS_COUNT;
  private static final int EXP_COUNT = 100;
  private static final int HOTELS_COUNT = 100;
  private static final int KEYS_COUNT = 2 * HOTELS_COUNT;
  private static final double EPS = 1e-8;

  private final VecBuilder value = new VecBuilder();
  private final ExecutorService executor = Executors.newFixedThreadPool(WORKERS_COUNT);

  // TODO: check & improve multithreading
  // TODO: replace HashMap with fast HashMap
  // TODO: replace int[] with SparseVec
  private final Map<Integer, AtomicIntegerArray> users;
  private final AtomicIntegerArray hotelsTotal;
  private final AtomicIntegerArray[] hotels;

  private final AtomicInteger currentKey = new AtomicInteger();
  private final AtomicReference<AtomicIntegerArray> currentValues = new AtomicReference<>();

  // cache
  private double[] default_factor;
  private final Map<Integer, double[]> cache = new HashMap<>();

  // buffer
  private final double[] results = new double[EXP_COUNT];

  private final SampleTask[] tasks = new SampleTask[EXP_COUNT];
  private final List<Future<Double>> futures = new ArrayList<>(EXP_COUNT);

  static {
    WORKERS_COUNT = Runtime.getRuntime().availableProcessors();
  }

  public Factor(final Map<Integer, AtomicIntegerArray> users, final AtomicIntegerArray hotelsTotal, final AtomicIntegerArray[] hotels) {
    this.users = users;
    this.hotelsTotal = hotelsTotal;
    this.hotels = hotels;

    for (int i = 0; i < KEYS_COUNT; ++i) {
      final int[] buffer = new int[KEYS_COUNT];
      Arrays.fill(buffer, 1);

      hotels[i] = new AtomicIntegerArray(buffer);
      hotelsTotal.set(i, KEYS_COUNT);
    }

    for (int i = 0; i < EXP_COUNT; ++i) {
      tasks[i] = new SampleTask();
      futures.add(null);
    }
  }

  public Factor() {
    this(new HashMap<>(), new AtomicIntegerArray(KEYS_COUNT), new AtomicIntegerArray[KEYS_COUNT]);
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
    final ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(path + FILE_NAME)));
    final HashMap<Integer, AtomicIntegerArray> users = (HashMap<Integer, AtomicIntegerArray>) in.readObject();
    final AtomicIntegerArray hotelsTotal = (AtomicIntegerArray) in.readObject();
    final AtomicIntegerArray[] hotels = (AtomicIntegerArray[]) in.readObject();
    in.close();

    return new Factor(users, hotelsTotal, hotels);
  }

  public double getFactor(final int user, final int hotel) {
    if (!users.containsKey(user)) {
      return getDefaultFactor()[hotel];
    }

    final double[] values = getCache(user);

    if (values[hotel] < EPS) {
      values[hotel] = compute(user, hotel, 1);
    }

    return values[hotel];
  }

  public void addFactor(final int user, final int hotel) {
    value.append(getFactor(user, hotel));
  }

  public void addEvent(final int user, final int hotel, final int hasBooked) {
    value.append(compute(user, hotel, hasBooked));
    update(user, hotel, hasBooked);
  }

  public void stop() {
    executor.shutdown();
  }

  private double compute(final int user, final int hotel, final int hasBooked) {
    currentKey.set(2 * hotel + hasBooked);
    currentValues.set(getUser(user));

    for (int taskIndex = 0; taskIndex < EXP_COUNT; ++taskIndex) {
      futures.set(taskIndex, executor.submit(tasks[taskIndex]));
    }

    try {
      for (int taskIndex = 0; taskIndex < EXP_COUNT; ++taskIndex) {
        results[taskIndex] = futures.get(taskIndex).get();
      }
    } catch (Exception e) {
      e.printStackTrace();
    }

    return getUCB(results);
  }

  private double[] getDefaultFactor() {
    if (default_factor == null) {
      default_factor = new double[HOTELS_COUNT];

      for (int hotel = 0; hotel < HOTELS_COUNT; ++hotel) {
        default_factor[hotel] = compute(-1, hotel, 1);
      }
    }

    return default_factor;
  }

  private void update(final int user, final int hotel, final int hasBooked) {
    final int key = 2 * hotel + hasBooked;
    final AtomicIntegerArray values = getUser(user);

    for (int i = 0; i < KEYS_COUNT; ++i) {
      if (values.get(i) > 1) {
        hotels[i].getAndIncrement(key);
        hotelsTotal.getAndIncrement(i);
      }
    }

    values.getAndIncrement(key);
  }

  private AtomicIntegerArray getUser(final int user) {
    AtomicIntegerArray values = users.get(user);

    if (values == null) {
      final int[] buffer = new int[KEYS_COUNT];
      Arrays.fill(buffer, 1);
      values = new AtomicIntegerArray(buffer);
      users.put(user, values);
    }

    return values;
  }

  private double[] getCache(final int user) {
    double[] values = cache.get(user);

    if (values == null) {
      values = new double[HOTELS_COUNT];
      cache.put(user, values);
    }

    return values;
  }

  private double getUCB(final double[] values) {
    final double average = Arrays.stream(values).average().orElse(0);
    final double std = Math.sqrt(Arrays.stream(values).map((x) -> Math.pow(x - average, 2)).average().orElse(0));
    return average + std;
  }

  private class SampleTask implements Callable<Double> {
    private final FastRandom random = new FastRandom();
    private final double[] dirichlet = new double[KEYS_COUNT];
    private final int[] params = new int[KEYS_COUNT];

    @Override
    public Double call() {
      final int key = currentKey.get();
      final AtomicIntegerArray values = currentValues.get();

      for (int i = 0; i < KEYS_COUNT; ++i) {
        params[i] = values.get(i);
      }

      random.nextDirichlet(params, dirichlet);

      double result = 0;
      for (int i = 0; i < KEYS_COUNT; ++i) {
        result += dirichlet[i] * random.nextBeta(hotels[i].get(key), hotelsTotal.get(i) - hotels[i].get(key));
      }

      return result;
    }
  }
}
