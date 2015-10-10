package com.spbsu.wiki;

import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Created by Юлиан on 10.10.2015.
 */
public class Rocchio {

    private HashMap<Integer, SparseVector> documents = new HashMap<>();

    private HashMap<Integer, SparseVector> conceptsWeights = new HashMap<>();

    private final TF tf;
    private final Object2IntOpenHashMap<String> indexes;
    private final Object2IntOpenHashMap<String> labelIndexes;

    private boolean allSequences = false;

    private Int2IntOpenHashMap overallDF = new Int2IntOpenHashMap();

    private static HashSet<String> used = new HashSet<>();


    private boolean builded = false;

    private int N = 0;

    public void setUsingAllSequances(boolean use){
        allSequences = use;
    }

    public Rocchio(TF tf){
        this.tf = tf;
        indexes = new Object2IntOpenHashMap<>();
        labelIndexes = new Object2IntOpenHashMap<>();
    }

    public String classify(String document) throws Exception{
        Map<String, Integer> map = allSequences ? tf.mapOfTheStringAllSequences(document.toLowerCase()) : tf.mapOfTheString(document.toLowerCase());
        for(String term : map.keySet())
            if(term.contains(" "))
                used.add(term);
        return classify(toVector(map));
    }

    public void addDocument(String document, String label){
        if(!labelIndexes.containsKey(label))
            labelIndexes.put(label, labelIndexes.size() + 1);
        addDocument(document, new int[]{labelIndexes.getInt(label)});
    }

    public void addDocument(String document, int label){
        addDocument(document, new int[]{label});
    }

    public void addDocument(String document, int[] labels){
        Map<String, Integer> map = allSequences ? tf.mapOfTheStringAllSequences(document.toLowerCase()) : tf.mapOfTheString(document.toLowerCase());

        for(String term : map.keySet())
            if(term.contains(" "))
                used.add(term);

        SparseVector vector = toVector(map);

        for(int label : labels) {
            if (!documents.containsKey(label))
                documents.put(label, new SparseVector());
            documents.get(label).add(vector);
        }
        for(int i : vector.keySet()){
            overallDF.addTo(i, 1);
        }
        N++;
    }


    public void buildClassifier(){
        for(int key : documents.keySet()){
            SparseVector concept = new SparseVector();
            SparseVector vector = documents.get(key);
            for(int i : vector.keySet()){
                concept.put(i, Math.log(vector.getValue(i) + 1) * Math.log(1.0 * N / overallDF.get(i)));
            }
            concept.normalize();
            conceptsWeights.put(key, concept);
        }
        builded = true;
    }

    public String classify(SparseVector vector){
        if(!builded)
            buildClassifier();
        SparseVector newVector = new SparseVector();
        for(int i : vector.keySet()){
            if(overallDF.get(i) > 0)
                newVector.put(i,Math.log(vector.getValue(i) + 1)*Math.log(1.0 * N / overallDF.get(i)));
        }
        newVector.normalize();
        Int2DoubleOpenHashMap angles = new Int2DoubleOpenHashMap();
        for(int label : conceptsWeights.keySet()){
            angles.put(label, Math.cos(newVector.angle(conceptsWeights.get(label))));
        }

        LinkedHashMap<Integer, Double> sorted = WikiUtils.sortByValues(angles, true);

        double sum = WikiUtils.sumOfElements(sorted.values());
        double currsum = 0;

        IntArrayList result = new IntArrayList();

        for(int key : sorted.keySet()) {
            result.add(key);
            currsum += sorted.get(key);
            if(result.size() == 1 || currsum/sum > 0.8)
                break;
        }
        for(int i : result)
            for(String key : labelIndexes.keySet())
                if(labelIndexes.getInt(key) == i)
                    return key;
        return "";
    }

    public int usedSize(){
        return used.size();
    }

    private SparseVector toVector(Map<String, Integer> mapOfDoc){
        for(String i : mapOfDoc.keySet())
            if(!indexes.containsKey(i))
                indexes.put(i, indexes.size() + 1);
        Int2DoubleOpenHashMap map = new Int2DoubleOpenHashMap();
        for(String key : mapOfDoc.keySet()){
            map.put(indexes.getInt(key), 1);
        }
        return new SparseVector(map);
    }

}
