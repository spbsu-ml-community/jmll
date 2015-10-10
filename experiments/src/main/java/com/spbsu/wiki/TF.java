package com.spbsu.wiki;

import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by Юлиан on 10.10.2015.
 */
public class TF {

    private final Object2IntOpenHashMap<String> tf;
    private final HashMap<String, HashSet<String>> sequences;

    public TF(){
        tf = new Object2IntOpenHashMap<>();
        sequences = new HashMap<>();
    }

    public TF(Set<String> dictionary){
        tf = new Object2IntOpenHashMap<>(dictionary.size());
        sequences = new HashMap<>();
        for(String i : dictionary)
            tf.put(i.toLowerCase(), 0);
        createSequences(dictionary);
    }

    private void createSequences(Set<String> dictionary){
        for(String term : dictionary){
            String[] seq = term.split(" ");
            if(seq.length > 1){
                String key = seq[0];
                if(!sequences.containsKey(key))
                    sequences.put(key, new HashSet<>());
                StringBuilder ngramm = new StringBuilder().append(key);
                for(int i = 1; i < seq.length; i++){
                    ngramm.append(" ").append(seq[i]);
                    sequences.get(key).add(ngramm.toString());
                }
            }
        }
    }

    public Object2IntOpenHashMap<String> mapOfTheString(String str){
        //returns the map where map.get(term) equals
        //the number of occurrences the term in the text
        return processString(str.toLowerCase(), new Object2IntOpenHashMap<>());
    }

    public Object2IntOpenHashMap<String> mapOfTheStringOnlyFromDict(String str){
        //returns the map where map.get(term) equals
        //the number of occurrences the term in the text
        //and these terms are from dictionary
        return processStringFromDict(str.toLowerCase(), new Object2IntOpenHashMap<>());
    }

    public void addString(String str){
        processString(str, this.tf);
    }

    private Object2IntOpenHashMap<String> processString(String str, Object2IntOpenHashMap<String> tf){
        String[] text = str.split(" ");
        for(int i = 0; i < text.length;){
            int index = findSequence(i, text);
            StringBuilder term = new StringBuilder().append(text[i]);
            for(i++; i < index; i++)
                term.append(" ").append(text[i]);
            tf.addTo(term.toString(), 1);
        }
        return tf;
    }

    private Object2IntOpenHashMap<String> processStringFromDict(String str, Object2IntOpenHashMap<String> tf){
        String[] text = str.split(" ");
        for(int i = 0; i < text.length;){
            int index = findSequence(i, text);
            StringBuilder term = new StringBuilder().append(text[i]);
            for(i++; i < index; i++)
                term.append(" ").append(text[i]);
            if(this.tf.containsKey(term.toString()))
                tf.addTo(term.toString(), 1);
        }
        return tf;
    }

    public Object2IntOpenHashMap<String> mapOfTheStringAllSequences(String str){
        Object2IntOpenHashMap<String> result = new Object2IntOpenHashMap<>();
        String[] text = str.split(" ");
        for(int i = 0; i < text.length; i++){
            for(String seq : findAllSequences(i, text)){
                result.addTo(seq, 1);
            }
        }
        return result;
    }

    private int findSequence(int index, String[] text){
        int lastIndex = index + 1;
        if(sequences.containsKey(text[index])){
            String key = text[index];
            StringBuilder term = new StringBuilder().append(key);
            for(int i = index + 1; i < text.length; i++){
                term.append(" ").append(text[i]);
                if(tf.containsKey(term.toString()))
                    lastIndex = i + 1;
                if(!sequences.get(key).contains(term.toString()))
                    break;
            }
        }
        return lastIndex;
    }

    private HashSet<String> findAllSequences(int index, String[] text){
        HashSet<String> result = new HashSet<>();
        if(sequences.containsKey(text[index])){
            String key = text[index];
            StringBuilder term = new StringBuilder().append(key);
            for(int i = index + 1; i < text.length; i++){
                term.append(" ").append(text[i]);
                if(tf.containsKey(term.toString()))
                    result.add(term.toString());
                if(!sequences.get(key).contains(term.toString()))
                    break;
            }
        }
        if(result.size() == 0)
            result.add(text[index]);
        return result;
    }

    public Object2IntOpenHashMap<String> getTf() {
        return tf;
    }

}
