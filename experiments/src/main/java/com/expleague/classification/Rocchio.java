package com.expleague.classification;

import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;

import java.util.*;
import java.util.function.Function;

/**
 * Метод классификации, описанный в
 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.447.7300&rep=rep1&type=pdf
 * <p/>
 * Для каждого класса считает tf idf слов.
 * Для входного документа также строит вектор с весами tf idf.
 * <p/>
 * Выбирается тот класс, который имеет наименьший угол с дынным документом.
 */
public class Rocchio {

  private HashMap<Integer, SparseVec> classes = new HashMap<>(); //для каждого класса содержит tf
  // слов

  private HashMap<Integer, SparseVec> conceptsWeights = new HashMap<>();//для каждого класса
  // содержит вес слов

  private final Function<String, String[]> splitter;
  private final Object2IntOpenHashMap<String> indexesOfTerms; //слово -> id
  private final Object2IntOpenHashMap<String> indexesOfClasses;// класс -> id

  private Int2IntOpenHashMap df = new Int2IntOpenHashMap();//DF слов

  private static HashSet<String> used = new HashSet<>();//какие фразы были использованы
  //можно заменить на ArrayList и получить число вхождений

  private boolean builded = false; //был ли построен классификатор

  private int N = 0; //число документов

  public Rocchio(Function<String, String[]> splitter) {
    this.splitter = splitter;
    indexesOfTerms = new Object2IntOpenHashMap<>();
    indexesOfClasses = new Object2IntOpenHashMap<>();
  }


  /**
   * Метод переводит документ из строки в вектор
   * и возвращает результат классификации
   * вектора.
   *
   * @param document - документ для классификации
   * @return результат классификации вектора
   */

  public String classify(String document) {
    String[] splited = splitter.apply(document.toLowerCase());
    Map<String, Integer> map = mapOfDocument(splited);
    for (String term : map.keySet())
      if (term.contains(" "))
        used.add(term);
    return classify(toVector(map));
  }

  /**
   * Дообавляет документ в обучающее множество.
   *
   * @param document - документ
   * @param label    - класс, к которому принадлежит документ
   */
  public void addDocument(String document, String label) {
    if (!indexesOfClasses.containsKey(label))
      indexesOfClasses.put(label, indexesOfClasses.size() + 1);
    addDocument(document, new int[]{indexesOfClasses.getInt(label)});
  }

  /**
   * Дообавляет документ в обучающее множество.
   *
   * @param document - документ
   * @param label    - класс, к которому принадлежит документ
   */
  public void addDocument(String document, int label) {
    addDocument(document, new int[]{label});
  }


  /**
   * Дообавляет документ в обучающее множество.
   * <p/>
   * Считает tf для каждого слова из документа и добавляет
   * это к количеству вхождений по классам.
   *
   * @param document - документ
   * @param labels   - классы, к которым принадлежит документ
   */

  public void addDocument(String document, int[] labels) {

    String[] splited = splitter.apply(document.toLowerCase());
    Map<String, Integer> map = mapOfDocument(splited);

    for (String term : map.keySet())
      if (term.contains(" "))
        used.add(term);

    SparseVec vector = toVector(map);

    for (int label : labels) {
      if (!classes.containsKey(label))
        classes.put(label, new SparseVec(0));
      classes.put(label, VecTools.sum(classes.get(label), vector));
    }
    for (int i : vector.indices.toArray()) {
      df.addTo(i, 1);
    }
    N++;
  }

  /**
   * Главная фаза обучения. Для каждой пары слово - класс
   * считает вес слова в данном классе.
   * <p/>
   * Вызывается автоматически, если метод классификации
   * до построения классификатора.
   */
  public void buildClassifier() {
    for (int key : classes.keySet()) {
      Int2DoubleOpenHashMap concept = new Int2DoubleOpenHashMap();
      SparseVec vector = classes.get(key);
      for (int i : vector.indices.toArray()) {
        concept.put(i, Math.log(vector.get(i) + 1) * Math.log(1.0 * N / df.get(i)));
      }
      conceptsWeights.put(key, convertMapToVec(concept));
    }
    builded = true;
  }

  /**
   * Метод классификации векторов.
   * <p/>
   * Входной вектор переводится в tf idf, после чего
   * считаются косинусы углов от текущего документа
   * ко всем остальным.
   * <p/>
   * Возвращает название класса, косинус с которым
   * наибольший. Если название отсутствует - возвращает
   * индекс класса.
   *
   * @param vector - документ для классификации
   * @return название или индекс класса документа
   */

  public String classify(SparseVec vector) {
    if (!builded)
      buildClassifier();

    //переводим вектор частот в вектор весов
    Int2DoubleOpenHashMap mapOfNewVector = new Int2DoubleOpenHashMap();
    for (int i : vector.indices.toArray()) {
      if (df.get(i) > 0)
        mapOfNewVector.put(i, Math.log(vector.get(i) + 1) * Math.log(1.0 * N / df.get(i)));
    }
    SparseVec newVector = convertMapToVec(mapOfNewVector);

    //Считаем косинусы
    Int2DoubleOpenHashMap cosines = new Int2DoubleOpenHashMap();
    for (int label : conceptsWeights.keySet()) {
      cosines.put(label, VecTools.cosine(newVector, conceptsWeights.get(label)));
    }

    //сортируем массив, находим ближайший класс, возвращаем
    LinkedHashMap<Integer, Double> sorted = sortByValues(cosines, true);

    int result = sorted.keySet().iterator().next();
    for (String key : indexesOfClasses.keySet())
      if (indexesOfClasses.getInt(key) == result)
        return key;
    return String.valueOf(result);
  }

  public static int usedSize() {
    return used.size();
  }

  /**
   * Переводит массив строк в map, где каждое
   * слово отображается в количество вхождений
   * в массив.
   *
   * @param terms - массив слов
   * @return map с частотами
   */

  private Map<String, Integer> mapOfDocument(String[] terms) {
    Object2IntOpenHashMap<String> result = new Object2IntOpenHashMap<>();
    for (String term : terms)
      result.addTo(term, 1);
    return result;
  }

  /**
   * Переводит map с частотами в вектор, переводя
   * каждый String в соответствующий index
   *
   * @param mapOfDoc - map с частотами
   * @return вектор частот
   */

  private SparseVec toVector(Map<String, Integer> mapOfDoc) {
    for (String i : mapOfDoc.keySet())
      if (!indexesOfTerms.containsKey(i))
        indexesOfTerms.put(i, indexesOfTerms.size() + 1);
    Int2DoubleOpenHashMap map = new Int2DoubleOpenHashMap();
    for (String key : mapOfDoc.keySet()) {
      map.put(indexesOfTerms.getInt(key), mapOfDoc.get(key));
    }
    return convertMapToVec(map);
  }

  public static SparseVec convertMapToVec(Map<Integer, Double> map) {
    IntArrayList indeces = new IntArrayList();
    DoubleArrayList values = new DoubleArrayList();
    for (int key : map.keySet()) {
      indeces.add(key);
      values.add(map.get(key));
    }
    return new SparseVec(0, indeces.toIntArray(), values.toDoubleArray()); //Is 0 allowed?
  }

  public static <K, V extends Comparable> LinkedHashMap<K, V> sortByValues(Map<K, V> map) {
    return sortByValues(map, false);
  }

  public static <K, V extends Comparable> LinkedHashMap<K, V> sortByValues(Map<K, V> map, boolean
      inverse) {
    LinkedHashMap<K, V> result = new LinkedHashMap<>();
    HashMap<V, ArrayList<K>> inv = new HashMap<>();
    for (K key : map.keySet()) {
      V value = map.get(key);
      if (!inv.containsKey(value))
        inv.put(value, new ArrayList<K>());
      inv.get(value).add(key);
    }
    ArrayList<V> values = new ArrayList<>(new HashSet<>(map.values()));
    if (inverse)
      Collections.sort(values, Collections.reverseOrder());
    else
      Collections.sort(values);
    for (V value : values) {
      for (K key : inv.get(value))
        result.put(key, value);
    }
    return result;
  }

}