package com.spbsu.wiki;

import com.spbsu.commons.func.Processor;
import com.spbsu.commons.io.codec.IntExpansion;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.seq.Seq;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;

import java.io.*;
import java.util.*;

/**
 * Created by Юлиан on 05.10.2015.
 */
public class WikiUtils {

    private static IntExpansion coding;

    private WikiUtils(){

    }

    public static byte[] intToByteArray(int value){
        byte[] result = new byte[4];
        for(int i = 0; i < 4; i++)
            result[i] = (byte)(value >> i*8);
        return result;
    }

    public static int[] byteArrayToIntArray(byte[] array){
        int[] result = new int[array.length/4];
        for(int i = 0; i < array.length/4; i++)
            for(int j = 0; j < 4; j++){
                result[i] = result[i] | (int)(array[4*i + j] & 0xff) << 8*j; // It's not redundant!!!
            }
        return result;
    }

    public static byte[] intArrayToByteArray(int[] array){
        byte[] result = new byte[array.length*4];
        for(int i = 0; i < array.length; i++){
            byte[] bytes = intToByteArray(array[i]);
            System.arraycopy(bytes, 0, result, i*4, 4);
        }
        return result;
    }

    public static String[] splitIntoSentences(String text){
        ArrayList<String> sentences = new ArrayList<>();
        sentences.add(text.replaceAll("\\.\"", "\\.\" "));
        ArrayList<String> result = new ArrayList<>();
        String[] separators = new String[]{". ", ".\" ", ".” "};
        for(String sep : separators) {
            for(String str : sentences) {
                int index = str.indexOf(sep);
                while (index != -1) {
                    String supposition = str.substring(0, index);
                    if (checkSentence(supposition)) {
                        result.add(supposition);
                        str = str.substring(index + 2);
                        index = str.indexOf(sep);
                    } else {
                        index = str.indexOf(sep, index + 2);
                    }
                }
                if(str.endsWith("."))
                    result.add(str.substring(0, str.length() - 1));
                else
                    result.add(str);
            }
            sentences = result;
            result = new ArrayList<>();
        }

        for(String str : sentences) {
            int index = str.indexOf(".");
            while(index != -1) {
                if (str.substring(0, index).matches(".*[a-zA-Z]{3,}\\)?") &&
                        str.substring(index + 1).matches("[a-zA-Z]+.*") &&
                        !str.substring(index + 1).endsWith("com") &&
                        !str.substring(index + 1).endsWith("net") &&
                        !str.substring(index + 1).endsWith("de") &&
                        !str.substring(index + 1).endsWith("com") &&
                        !str.substring(index + 1).endsWith("exe") &&
                        !str.substring(index + 1).endsWith("jpg")) {
                    result.add(str.substring(0, index).trim());
                    str = str.substring(index + 1);
                }
                index = str.indexOf(".", index + 1);
            }
            result.add(str.trim());
        }
        return result.toArray(new String[result.size()]);
    }

    private static boolean checkSentence(String sentence){
        return sentence.length() > 10 &&
                !sentence.endsWith(" e.i") &&
                !sentence.endsWith(" (b") &&
                !sentence.endsWith(" (d") &&
                !sentence.endsWith(" (cf") &&
                !sentence.endsWith(" e.g") &&
                !sentence.endsWith(" etc") &&
                !sentence.endsWith(" U.S") &&
                !sentence.endsWith(" Jr") &&
                !sentence.endsWith(" Mr") &&
                !sentence.endsWith(" Ms") &&
                !sentence.endsWith(" St") &&
                !sentence.endsWith(" Dr") &&
                !sentence.endsWith(" Co") &&
                !sentence.endsWith(" Inc") &&
                sentence.charAt(sentence.length() - 2) != '.' &&
                sentence.charAt(sentence.length() - 2) != ' ';
    }

    public static IntSeq stringToIntSeq(String str){
        return StringToIntSeq.convert(str);
    }

    public static String removePunctuation(String str){
        if(str.length() > 1 && str.charAt(str.length() - 1) == '.')
            str = str.substring(0, str.length() - 1);
        str = str.replaceAll("“", "").replaceAll("”", "");
        str = str.replaceAll("\\([^()].*?\\)", " ");
        str = str.replaceAll("[,!?\"/\\\\]", " ");
        return str.replaceAll("\\s+", " ").trim();
    }

    public static Map<String, Integer> getIndexes(){
        return StringToIntSeq.indexes;
    }

    public static Processor<String> getDefaultProcessor(){
        return (String text) -> {String[] sentences = WikiUtils.splitIntoSentences(text);
            for(String sentence : sentences){
                sentence = removePunctuation(sentence);
                sentence = sentence.toLowerCase();
                IntSeq seq = WikiUtils.stringToIntSeq(sentence);
                coding.accept(seq);
            }
        };
    }

    public static void readIndexes(File file) throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        StringToIntSeq.clearIndexes();
        int maxIndex = 0;
        while((line = reader.readLine()) != null){
            int index = Integer.parseInt(line.split("\t")[1]);
            if(index > maxIndex){
                maxIndex = index;
                while(StringToIntSeq.reverseIndex.size() < maxIndex + 1){
                    StringToIntSeq.reverseIndex.add(null);
                }
            }
            String word = line.split("\t")[0];
//            StringToIntSeq.indexes.put(word, index);
            StringToIntSeq.reverseIndex.set(index, word);
        }
    }

    public static void writeIndexesInFile(File file) throws IOException{
        FileWriter fw = new FileWriter(file);
        Map<String, Integer> indexes = getIndexes();
        for(String key : indexes.keySet()){
            fw.write(key + "\t" + indexes.get(key) + "\n");
        }
        fw.close();
    }

    public static void setDefaultCoding(IntExpansion expansion){
        coding = expansion;
    }

    public static Map<String, Integer> getWordsDictionary(){
        Object2IntOpenHashMap<String> result = new Object2IntOpenHashMap<>();
        for (int j = 0; j < getCurrentDictionary().size(); j++) {
            StringBuilder sb = new StringBuilder();
            Seq seq = getCurrentDictionary().alphabet().get(j);
            if (seq.length() < 2)
                continue;
            for (int t = 0; t < seq.length(); t++)
                sb.append(StringToIntSeq.reverseIndex.get((int)seq.at(t))).append(" ");
            result.put(sb.toString(), coding.expansion().resultFreqs()[j]);
        }
        return result;
    }

    public static String intSeqToWords(int ... integers){
        StringBuilder result = new StringBuilder();
        for(int i : integers){
            result.append(StringToIntSeq.reverseIndex.get(i)).append(" ");
        }
        return result.toString().trim();
    }

    public static <K, V extends Comparable> LinkedHashMap<K, V> sortByValues(Map<K, V> map){
        return sortByValues(map, false);
    }

    public static <K, V extends Comparable> LinkedHashMap<K, V> sortByValues(Map<K, V> map, boolean inverse){
        LinkedHashMap<K, V> result = new LinkedHashMap<>();
        HashMap<V, ArrayList<K>> inv = new HashMap<>();
        for(K key : map.keySet()){
            V value = map.get(key);
            if(!inv.containsKey(value))
                inv.put(value, new ArrayList<K>());
            inv.get(value).add(key);
        }
        ArrayList<V> values = new ArrayList<>(new HashSet<>(map.values()));
        if(inverse)
            Collections.sort(values, Collections.reverseOrder());
        else
            Collections.sort(values);
        for(V value : values){
            for(K key : inv.get(value))
                result.put(key, value);
        }
        return result;
    }

    public static Dictionary<Integer> getCurrentDictionary(){
        return coding.expansion().result();
    }

    public static Set<String> readDictionaryFromFile(File file) throws IOException{
        HashSet<String> dictionary = new HashSet<>();
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            dictionary.add(line.split("\t")[0]);
        }
        return dictionary;
    }

    public static <T extends Number> double sumOfElements(Collection<T> collection){
        double res = 0;
        for(T element : collection){
            res += element.doubleValue();
        }
        return res;
    }

    private static class StringToIntSeq {

        private final static Object2IntOpenHashMap<String> indexes = new Object2IntOpenHashMap<>();
        private final static ArrayList<String> reverseIndex = new ArrayList<>();

        public static IntSeq convert(String str){
            IntArrayList result = new IntArrayList();
            for(String s : str.split(" ")){
                if(!indexes.containsKey(s)) {
                    indexes.put(s, indexes.size() + 1);
                    while(reverseIndex.size() < indexes.size() + 1) {
                        reverseIndex.add(null);
                    }
                    reverseIndex.set(indexes.size(), s);
                }
                result.add(indexes.getInt(s));

            }
            return new IntSeq(result.toIntArray());
        }

        public Object2IntOpenHashMap<String> getIndexes(){
            return indexes;
        }

        public static void clearIndexes(){
            reverseIndex.clear();
            indexes.clear();
        }

    }

}
