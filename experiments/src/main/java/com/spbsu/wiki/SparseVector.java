package com.spbsu.wiki;

import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;


public class SparseVector extends Int2DoubleOpenHashMap{

    private double norm = 1.0;
    private boolean normalized = false;

    public SparseVector(){
        super();
    }

    public SparseVector(Int2DoubleOpenHashMap map){
        super(map);
        normalize();
    }

    public double getNorm(){
        return norm;
    }

    public double getValue(int index){
        return get(index)*norm;
    }

    public double angle(SparseVector a){
        return angle(this, a);
    }

    public double dotProduct(SparseVector a){
        return dotProduct(this, a);
    }

    private static double dotProduct(SparseVector a, SparseVector b){
        if(!a.normalized)
            a.normalize();
        if(!b.normalized)
            b.normalize();
        IntOpenHashSet set = new IntOpenHashSet(a.keySet());
        set.retainAll(b.keySet());
        double result = 0.0;
        for(int i : set){
            result += a.get(i)*b.get(i);
        }
        return result;
    }

    private static double angle(SparseVector a, SparseVector b) {
        return Math.acos(dotProduct(a, b));
    }

    public void normalize(){
        double newNorm = 0.0;
        for(int i : keySet()){
            newNorm+= Math.pow(getValue(i),2);
        }
        newNorm = Math.sqrt(newNorm);
        for(int i : keySet()){
            put(i, getValue(i) / newNorm);
        }
        norm = newNorm;
        normalized = true;
    }



    public SparseVector add(SparseVector vect){
        IntOpenHashSet set = new IntOpenHashSet(this.keySet());
        set.addAll(vect.keySet());
        for(int i : set){
            this.put(i, this.getValue(i) + vect.getValue(i));
        }
        norm = 1;
        normalize();
        return this;
    }

    public int crossingSize(SparseVector vect){
        return crossingSize(this, vect);
    }

    public int crossingSize(SparseVector vect, double num){
        return crossingSize(this, vect, num);
    }

    private static int crossingSize(SparseVector a, SparseVector b){
        IntOpenHashSet set = new IntOpenHashSet(a.keySet());
        set.retainAll(b.keySet());
        return set.size();
    }

    private static int crossingSize(SparseVector a, SparseVector b, double num){
        int count = 0;
        for(Integer key : a.keySet()){
            if(b.keySet().contains(key) && a.getValue(key) >= num){
                count++;
            }
        }
        return count;
    }

}
