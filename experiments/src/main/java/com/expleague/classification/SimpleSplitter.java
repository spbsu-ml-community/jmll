package com.expleague.classification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * Разбивает строку на слова и фразы из заданного словаря.
 * <p/>
 * При разбиении использует жадный алгоритм, на каждом
 * шаге выбирая фразу наибольшей длины, с которой начинается
 * текущая строка. Если соварь не содержит подходящей фразы,
 * то берется первое слово. Далее строка урезается в соответствии
 * с найденной фразой/словом.
 * <p/>
 * Если использован конструктор SimpleSplitter(Set<String> dictionary),
 * то фразы берутся из dictionary.
 * <p/>
 * Если словарь не задан, то текст разбивается на слова.
 */
public class SimpleSplitter {

  private final static String delimiter = "";
  private final HashSet<String> dictionary;
  private final HashMap<String, HashSet<String>> sequences;

  public final HashSet<String> used = new HashSet<>();

  public SimpleSplitter() {
    dictionary = new HashSet<>();
    sequences = new HashMap<>();
  }

  public SimpleSplitter(Set<String> dictionary) {
    this.dictionary = new HashSet<>();
    sequences = new HashMap<>();
    for (String i : dictionary)
      this.dictionary.add(i.toLowerCase());
    createSequences(dictionary);
  }


  /**
   * Разбивает входную строку на фразы и слова
   *
   * @param str - входная строка
   * @return разбитый текст
   */
  public String[] split(String str) {
    String[] text = str.split(delimiter);
    ArrayList<String> result = new ArrayList<>();
    for (int i = 0; i < text.length; ) {
      int index = findSequence(i, text);
      StringBuilder term = new StringBuilder().append(text[i]);
      for (i++; i < index; i++)
        term.append(delimiter).append(text[i]);
      result.add(term.toString());
    }
    String[] res = new String[result.size()];
    result.toArray(res);
    return res;
  }


  /**
   * Заполняет объект HashMap<String, HashSet<String>> sequences.
   * <p/>
   * Из каждой фразы словаря извлекается первое слово.
   * Это слово отображается на все возможные
   * подстроки фразы, начинающиеся с первого слова.
   * <p/>
   * Структура используется для ускорения поиска подфраз.
   *
   * @param dictionary
   */

  private void createSequences(Set<String> dictionary) {
    for (String term : dictionary) {
      String[] seq = term.split(delimiter);
      if (seq.length > 1) {
        String key = seq[0];
        if (!sequences.containsKey(key))
          sequences.put(key, new HashSet<>());
        StringBuilder ngramm = new StringBuilder().append(key);
        for (int i = 1; i < seq.length; i++) {
          ngramm.append(delimiter).append(seq[i]);
          sequences.get(key).add(ngramm.toString());
        }
      }
    }
  }


  /**
   * Ищет фразу наибольшей длины, которая начинается с указанного индекса.
   *
   * @param index - индекс, с которого начинается поиск
   * @param text  - строка, разбитая на слова
   * @return индекс последнего слова найденной фразы
   */
  private int findSequence(int index, String[] text) {
    int lastIndex = index + 1;
    if (sequences.containsKey(text[index])) {
      String key = text[index];
      StringBuilder term = new StringBuilder().append(key);
      for (int i = index + 1; i < text.length; i++) {
        term.append(delimiter).append(text[i]);
        if (dictionary.contains(term.toString()))
          lastIndex = i + 1;
        if (!sequences.get(key).contains(term.toString()))
          break;
      }
    }
    return lastIndex;
  }


}