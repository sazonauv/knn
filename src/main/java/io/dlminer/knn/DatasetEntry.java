package io.dlminer.knn;

public class DatasetEntry {

    public int id;

    public Double[] vector;

    public double time;

    public int label;

    public DatasetEntry() {
        super();
    }

    public boolean isTrivial() {
        for (int i=0; i<vector.length; i++) {
            if (vector[i] > 0) {
                return false;
            }
        }
        return true;
    }

    public static boolean isTrivial(Double[] vector) {
        for (int i=0; i<vector.length; i++) {
            if (vector[i] > 0) {
                return false;
            }
        }
        return true;
    }


}
