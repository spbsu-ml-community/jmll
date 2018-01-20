package com.expleague.classification;

import com.expleague.commons.seq.CharSeqAdapter;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;

import java.util.*;


/**
 * Разбивает строку на слова и фразы из заданного словаря.
 * <p/>
 * Строка разбивается так, чтобы сумма весов найденных фраз
 * была максимальной. В качестве весов берутся частоты
 * фраз в исходной коллекции, на которой фразы построены.
 * <p/>
 * Если использован конструктор WeighedSplitter(Map<String, Double> weights),
 * то фразы берутся из weights.
 * <p/>
 * Если словарь не задан, то текст просто разбивается на слова.
 */
public class WeighedSplitter {

  private final static String delimiter = "";
  private final HashMap<String, Object2DoubleOpenHashMap<String>> phrases; //last word -> phrase
    // -> weight; для ускорения
  private final static double Z = 0; //вес, который имеет одно слово

  public WeighedSplitter() {
    phrases = new HashMap<>();
  }

  public WeighedSplitter(Map<String, Double> weights) {
    phrases = new HashMap<>();
    for (String term : weights.keySet()) {
      String last = twoLastWords(term);
      if (!phrases.containsKey(last))
        phrases.put(last, new Object2DoubleOpenHashMap<>());
      phrases.get(last).put(term, weights.get(term));
    }
  }

  /**
   * Разбивает входную строку на фразы и слова
   *
   * @param str - входная строка
   * @return разбитый текст
   */

  public String[] split(String str) {
    CharSequence words = new CharSeqAdapter(str);
    double[] weights = new double[words.length() + 1]; //i-ый элемент содержит максимальный вес,
      // который можно
    //получить для str.substring(0, i)
    int[] prevs = new int[words.length() + 1]; //i-ый элемент содержит индекс, с которого начинается
    //последняя фраза при разбиении с максимальным весом
    prevs[0] = 0;
    for (int i = 0; i < words.length(); i++) {
      //в начале для каждой строки полагаем, что лучшее разбиение для substring(0, i + 1)
      //это лучшее разбиение для substring(0, i) плюс i-е слово.
      //далее пытаемся найти максимальное разбиение с помощью фраз.
      weights[i + 1] = weights[i] + Z;
      prevs[i + 1] = i;
      String last = twoLastWordsFast(words, i);
      if (last == null)
        continue;
      Map<String, Double> map = phrases.get(last);
      if (map != null) {
        for (String phrase : map.keySet()) {
          if (endsWith(words, i, phrase)) {
            int prev = i + 1 - phrase.split(delimiter).length;
            double weight = phrases.get(last).getDouble(phrase) + weights[prev];
            if (weights[i + 1] < weight) {
              weights[i + 1] = weight;
              prevs[i + 1] = prev;
            }
          }
        }
      }
    }
    //получаем конкретное разбиение,
    //восстанавливая слова с помощью prevs
    LinkedList<String> list = new LinkedList<>();
    int index = weights.length - 1;
    while (index > 0) {
      StringBuilder sb = new StringBuilder();
      int prev = prevs[index];
      sb.append(words.charAt(prev));
      while (++prev < index) {
        sb.append(delimiter).append(words.charAt(prev));
      }
      list.offerFirst(sb.toString());
      index = prevs[index];
    }
    String[] res = new String[list.size()];
    list.toArray(res);
    return res;
  }

  /**
   * Возвращает два последних слова строки.
   *
   * @param str - входная строка
   * @return строку с двумя последними словами
   */
  private static String twoLastWords(String str) {
    String[] strings = str.split(delimiter);
    if (strings.length == 1)
      return null;
    return new StringBuilder(strings[strings.length - 2]).append(delimiter).append
        (strings[strings.length - 1]).toString();
  }

  private static String twoLastWordsFast(String[] words, int index) {
    if (index == 0)
      return null;
    return new StringBuilder(words[index - 1]).append(delimiter).append(words[index]).toString();
  }

  public static String twoLastWordsFast(CharSequence words, int index) {
    if (index == 0)
      return null;
    //        return new StringBuilder(words.charAt(index - 1)).append(delimiter).append(words
      // .charAt(index)).toString();
    return new StringBuilder().append(words.charAt(index - 1)).append(words.charAt(index))
        .toString();
  }

  /**
   * Возвращает массив подстрок, начинающихся с первого слова.
   *
   * @param str - входная строка
   * @return массив всех подстрок
   */

  private static String[] substrings(String str) {
    String[] words = str.split(delimiter);
    String[] res = new String[words.length];
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < words.length; i++) {
      sb.append(words[i]).append(delimiter);
      res[i] = sb.toString().trim();
    }
    return res;
  }

  public static boolean endsWith(CharSequence sequence, int index, String phrase) {
    if (phrase.length() - 1 <= index) {
      for (int i = phrase.length() - 1; i >= 0; i--, index--) {
        if (sequence.charAt(index) != phrase.charAt(i))
          return false;
      }
      return true;
    }
    return false;
  }

}
